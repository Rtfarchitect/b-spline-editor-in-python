import igl
import numpy as np

import sys
import os

# Try different import strategies based on how PyIGL is installed
try:
    from iglhelpers import *
except ImportError:
    try:
        from igl.iglhelpers import *
    except ImportError:
        # If iglhelpers is not found, we'll define a simple version of p2e
        def p2e(point):
            return np.array([point[0], point[1], point[2]], dtype=np.float64)

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

class BSplineViewer:
    def __init__(self, width=800, height=600, title="B-Spline Editor"):
        # Initialize GLFW
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        # Create a window
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        # Make the window's context current
        glfw.make_context_current(self.window)

        # Set callbacks
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self.resize_callback)

        # Initialize OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Set up the camera
        self.camera_distance = 5.0
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0
        self.mouse_prev_x = 0.0
        self.mouse_prev_y = 0.0
        self.mouse_pressed = False
        
        # Initialize the B-spline data
        self.init_bspline()
        
    def init_bspline(self):
        # B-spline parameters
        self.degree = 3  # Cubic B-spline
        self.nu = 7  # Number of control points in u direction
        self.nv = 7  # Number of control points in v direction
        
        # Create a grid of control points
        u = np.linspace(-1, 1, self.nu)
        v = np.linspace(-1, 1, self.nv)
        
        self.control_points = np.zeros((self.nu * self.nv, 3))
        
        idx = 0
        for i in range(self.nu):
            for j in range(self.nv):
                self.control_points[idx] = [u[i], v[j], 0.0]
                idx += 1
        
        # Create a connectivity list for the control grid
        self.grid_edges = []
        for i in range(self.nu):
            for j in range(self.nv-1):
                self.grid_edges.append([i*self.nv + j, i*self.nv + j + 1])
        
        for j in range(self.nv):
            for i in range(self.nu-1):
                self.grid_edges.append([i*self.nv + j, (i+1)*self.nv + j])
        
        self.grid_edges = np.array(self.grid_edges)
        
        # Knot vectors
        self.u_knots = self.create_uniform_knots(self.nu, self.degree)
        self.v_knots = self.create_uniform_knots(self.nv, self.degree)
        
        # Rendering flags
        self.show_control_grid = True
        self.show_surface = True
        
        # Store patches
        self.patches = [self.control_points.copy()]
        self.active_patch = 0
        
        # Generate the surface mesh
        self.surface_v, self.surface_f = self.generate_surface_mesh()
        
        # Selected control point for dragging
        self.selected_point = None
        self.dragging = False
        
        # Print instructions
        print("B-Spline Editor Controls:")
        print("  Left-click and drag: Rotate view")
        print("  Right-click and drag: Move control points")
        print("  Scroll: Zoom in/out")
        print("  Key 1: Toggle control grid visibility")
        print("  Key 2: Toggle surface visibility")
        print("  Key A: Add new B-spline patch")
        print("  Key N: Switch to next patch")
        print("  Key P: Switch to previous patch")
        print("  Key ESC: Exit")
    
    def create_uniform_knots(self, n, k):
        # Create a uniform knot vector with multiplicity k+1 at the ends
        total_knots = n + k + 1
        knots = np.zeros(total_knots)
        
        # Set up multiplicity at the ends
        for i in range(k+1):
            knots[i] = 0.0
            knots[-(i+1)] = 1.0
        
        # Set up internal knots
        for i in range(k+1, total_knots - (k+1)):
            knots[i] = (i - k) / (n - k)
        
        return knots
    
    def basis_function(self, u, i, k, knots):
        # Evaluate the B-spline basis function of degree k at parameter u
        if k == 0:
            if knots[i] <= u < knots[i+1] or (abs(u - knots[i+1]) < 1e-10 and abs(knots[i+1] - knots[-1]) < 1e-10):
                return 1.0
            else:
                return 0.0
        
        d1 = knots[i+k] - knots[i]
        d2 = knots[i+k+1] - knots[i+1]
        
        f1 = 0.0 if d1 < 1e-10 else (u - knots[i]) / d1
        f2 = 0.0 if d2 < 1e-10 else (knots[i+k+1] - u) / d2
        
        return f1 * self.basis_function(u, i, k-1, knots) + f2 * self.basis_function(u, i+1, k-1, knots)
    
    def evaluate_surface_point(self, u, v):
        # Evaluate surface point at parameter (u, v)
        point = np.zeros(3)
        
        control_pts = self.patches[self.active_patch].reshape(self.nu, self.nv, 3)
        
        for i in range(self.nu):
            for j in range(self.nv):
                basis_u = self.basis_function(u, i, self.degree, self.u_knots)
                basis_v = self.basis_function(v, j, self.degree, self.v_knots)
                point += control_pts[i, j] * basis_u * basis_v
        
        return point
    
    def generate_surface_mesh(self, resolution=20):
        # Generate a surface mesh for visualization
        vertices = []
        faces = []
        
        u_steps = resolution
        v_steps = resolution
        
        # Create vertices
        for i in range(u_steps + 1):
            u = i / u_steps
            for j in range(v_steps + 1):
                v = j / v_steps
                vertices.append(self.evaluate_surface_point(u, v))
        
        # Create faces (triangulation)
        for i in range(u_steps):
            for j in range(v_steps):
                v0 = i * (v_steps + 1) + j
                v1 = v0 + 1
                v2 = (i + 1) * (v_steps + 1) + j
                v3 = v2 + 1
                
                # Add two triangles per grid cell
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])
        
        return np.array(vertices), np.array(faces)
    
    def add_new_patch(self):
        # Add a new patch by duplicating the current one and offsetting it
        new_patch = self.patches[self.active_patch].copy()
        new_patch[:, 2] += 0.5  # Offset in z-direction
        self.patches.append(new_patch)
        self.active_patch = len(self.patches) - 1
        self.surface_v, self.surface_f = self.generate_surface_mesh()
        print(f"Added new patch. Now editing patch {self.active_patch + 1} of {len(self.patches)}")
    
    def switch_patch(self, next_patch=True):
        # Switch to the next or previous patch
        if len(self.patches) > 1:
            if next_patch:
                self.active_patch = (self.active_patch + 1) % len(self.patches)
            else:
                self.active_patch = (self.active_patch - 1) % len(self.patches)
            
            self.surface_v, self.surface_f = self.generate_surface_mesh()
            print(f"Now editing patch {self.active_patch + 1} of {len(self.patches)}")
    
    def toggle_control_grid(self):
        self.show_control_grid = not self.show_control_grid
        print(f"Control grid {'visible' if self.show_control_grid else 'hidden'}")
    
    def toggle_surface(self):
        self.show_surface = not self.show_surface
        print(f"B-spline surface {'visible' if self.show_surface else 'hidden'}")
    
    def find_closest_control_point(self, x, y):
        # Convert screen coordinates to 3D ray
        width, height = glfw.get_window_size(self.window)
        
        # Get the viewport
        viewport = glGetIntegerv(GL_VIEWPORT)
        
        # Get the modelview and projection matrices
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Convert to NDC
        y = height - y  # Invert Y since OpenGL has origin at bottom-left
        
        # Unproject to get far and near points on the ray
        nearPos = gluUnProject(x, y, 0.0, modelview, projection, viewport)
        farPos = gluUnProject(x, y, 1.0, modelview, projection, viewport)
        
        # Calculate ray direction
        ray_dir = np.array(farPos) - np.array(nearPos)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Find closest control point to the ray
        control_pts = self.patches[self.active_patch]
        min_dist = float('inf')
        closest_idx = -1
        
        for i, point in enumerate(control_pts):
            # Calculate the distance from the point to the ray
            v = point - np.array(nearPos)
            dist = np.linalg.norm(np.cross(v, ray_dir))
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Only select if it's reasonably close
        if min_dist < 0.2:
            return closest_idx
        
        return None
    
    def move_control_point(self, x, y):
        if self.selected_point is None:
            return
        
        # Get window size
        width, height = glfw.get_window_size(self.window)
        
        # Get viewport
        viewport = glGetIntegerv(GL_VIEWPORT)
        
        # Get the modelview and projection matrices
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Convert to NDC
        y = height - y  # Invert Y
        
        # Get the current point's z-depth in screen space
        point = self.patches[self.active_patch][self.selected_point]
        winZ = gluProject(point[0], point[1], point[2], modelview, projection, viewport)[2]
        
        # Unproject the new mouse position to 3D
        new_pos = gluUnProject(x, y, winZ, modelview, projection, viewport)
        
        # Update the control point position
        self.patches[self.active_patch][self.selected_point] = np.array(new_pos)
        
        # Update the surface
        self.surface_v, self.surface_f = self.generate_surface_mesh()
    
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_1:
                self.toggle_control_grid()
            elif key == glfw.KEY_2:
                self.toggle_surface()
            elif key == glfw.KEY_A:
                self.add_new_patch()
            elif key == glfw.KEY_N:
                self.switch_patch(True)
            elif key == glfw.KEY_P:
                self.switch_patch(False)
    
    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.mouse_pressed = True
                x, y = glfw.get_cursor_pos(window)
                self.mouse_prev_x = x
                self.mouse_prev_y = y
            elif action == glfw.RELEASE:
                self.mouse_pressed = False
        
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                x, y = glfw.get_cursor_pos(window)
                self.selected_point = self.find_closest_control_point(x, y)
                if self.selected_point is not None:
                    self.dragging = True
            elif action == glfw.RELEASE:
                self.dragging = False
                self.selected_point = None
    
    def cursor_pos_callback(self, window, x, y):
        if self.mouse_pressed and not self.dragging:
            # Rotate view
            dx = x - self.mouse_prev_x
            dy = y - self.mouse_prev_y
            
            self.camera_angle_y += dx * 0.01
            self.camera_angle_x += dy * 0.01
            
            self.mouse_prev_x = x
            self.mouse_prev_y = y
        
        elif self.dragging and self.selected_point is not None:
            # Move control point
            self.move_control_point(x, y)
    
    def scroll_callback(self, window, xoffset, yoffset):
        # Zoom in/out
        self.camera_distance -= yoffset * 0.1
        if self.camera_distance < 0.1:
            self.camera_distance = 0.1
    
    def resize_callback(self, window, width, height):
        # Update viewport
        glViewport(0, 0, width, height)
    
    def render(self):
        # Clear the buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = glfw.get_window_size(self.window)
        aspect = width / height
        gluPerspective(45.0, aspect, 0.1, 100.0)
        
        # Set up modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Set up camera
        gluLookAt(0, 0, self.camera_distance,
                 0, 0, 0,
                 0, 1, 0)
        
        # Apply camera rotations
        glRotatef(self.camera_angle_x * 180.0 / np.pi, 1, 0, 0)
        glRotatef(self.camera_angle_y * 180.0 / np.pi, 0, 1, 0)
        
        # Set up lighting
        light_position = [1.0, 1.0, 1.0, 0.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        
        # Draw the control grid
        if self.show_control_grid:
            glDisable(GL_LIGHTING)
            
            # Draw control points
            glPointSize(5.0)
            glBegin(GL_POINTS)
            glColor3f(0.0, 0.0, 1.0)  # Blue
            for point in self.patches[self.active_patch]:
                glVertex3f(point[0], point[1], point[2])
            glEnd()
            
            # Draw grid lines
            glColor3f(1.0, 0.0, 0.0)  # Red
            glBegin(GL_LINES)
            for edge in self.grid_edges:
                p1 = self.patches[self.active_patch][edge[0]]
                p2 = self.patches[self.active_patch][edge[1]]
                glVertex3f(p1[0], p1[1], p1[2])
                glVertex3f(p2[0], p2[1], p2[2])
            glEnd()
            
            glEnable(GL_LIGHTING)
        
        # Draw the B-spline surface
        if self.show_surface:
            glEnable(GL_LIGHTING)
            
            # Set material properties
            mat_diffuse = [0.0, 0.8, 0.0, 1.0]  # Green
            mat_specular = [1.0, 1.0, 1.0, 1.0]
            mat_shininess = [50.0]
            
            glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
            glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
            glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)
            
            # Draw the surface triangles
            glBegin(GL_TRIANGLES)
            for face in self.surface_f:
                # Calculate face normal
                v0 = self.surface_v[face[0]]
                v1 = self.surface_v[face[1]]
                v2 = self.surface_v[face[2]]
                
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                glNormal3f(normal[0], normal[1], normal[2])
                glVertex3f(v0[0], v0[1], v0[2])
                glVertex3f(v1[0], v1[1], v1[2])
                glVertex3f(v2[0], v2[1], v2[2])
            glEnd()
    
    def run(self):
        # Main loop
        while not glfw.window_should_close(self.window):
            # Render
            self.render()
            
            # Swap buffers
            glfw.swap_buffers(self.window)
            
            # Poll for events
            glfw.poll_events()
        
        # Clean up
        glfw.terminate()

if __name__ == "__main__":
    try:
        viewer = BSplineViewer()
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        glfw.terminate()