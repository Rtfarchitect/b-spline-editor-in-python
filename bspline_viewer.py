import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

from bspline_math import (
    create_uniform_knots, 
    evaluate_surface_point, 
    generate_surface_mesh,
    create_control_grid
)
from gl_helpers import (
    setup_lighting, 
    setup_camera, 
    draw_control_grid, 
    draw_surface,
    find_closest_control_point,
    unproject_screen_coords
)

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
        setup_lighting()
        
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
        
        # Create initial control points and grid
        self.control_points, self.grid_edges = create_control_grid(self.nu, self.nv)
        
        # Knot vectors
        self.u_knots = create_uniform_knots(self.nu, self.degree)
        self.v_knots = create_uniform_knots(self.nv, self.degree)
        
        # Rendering flags
        self.show_control_grid = True
        self.show_surface = True
        
        # Store patches
        self.patches = [self.control_points.copy()]
        self.active_patch = 0
        
        # Generate the surface mesh
        self.update_surface()
        
        # Selected control point for dragging
        self.selected_point = None
        self.dragging = False
        
        # Print instructions
        self.print_instructions()
    
    def print_instructions(self):
        print("B-Spline Editor Controls:")
        print("  Left-click and drag: Rotate view")
        print("  Right-click and drag: Move control points")
        print("  Scroll: Zoom in/out")
        print("  Key 1: Toggle control grid visibility")
        print("  Key 2: Toggle surface visibility")
        print("  Key A: Add new B-spline patch (copy of current)")
        print("  Key N: Create new blank B-spline patch")
        print("  Key TAB: Switch to next patch")
        print("  Key P: Switch to previous patch")
        print("  Key ESC: Exit")
    
    def update_surface(self):
        """Update the surface mesh based on current control points."""
        control_pts = self.patches[self.active_patch]
        self.surface_v, self.surface_f = generate_surface_mesh(
            control_pts, 
            self.nu, 
            self.nv, 
            self.degree, 
            self.u_knots, 
            self.v_knots
        )
    
    def add_new_patch(self):
        """Add a new patch by duplicating the current one and offsetting it."""
        new_patch = self.patches[self.active_patch].copy()
        new_patch[:, 2] += 0.5  # Offset in z-direction
        self.patches.append(new_patch)
        self.active_patch = len(self.patches) - 1
        self.update_surface()
        print(f"Added new patch (copy of current). Now editing patch {self.active_patch + 1} of {len(self.patches)}")
    
    def create_new_patch(self):
        """Create a brand new patch from scratch."""
        # Create fresh control points for a new patch
        new_control_points, _ = create_control_grid(self.nu, self.nv)
        
        # Add it to the patches list
        self.patches.append(new_control_points)
        self.active_patch = len(self.patches) - 1
        self.update_surface()
        print(f"Created new blank patch. Now editing patch {self.active_patch + 1} of {len(self.patches)}")
    
    def switch_patch(self, next_patch=True):
        """Switch to the next or previous patch."""
        if len(self.patches) > 1:
            if next_patch:
                self.active_patch = (self.active_patch + 1) % len(self.patches)
            else:
                self.active_patch = (self.active_patch - 1) % len(self.patches)
            
            self.update_surface()
            print(f"Now editing patch {self.active_patch + 1} of {len(self.patches)}")
    
    def toggle_control_grid(self):
        """Toggle control grid visibility."""
        self.show_control_grid = not self.show_control_grid
        print(f"Control grid {'visible' if self.show_control_grid else 'hidden'}")
    
    def toggle_surface(self):
        """Toggle B-spline surface visibility."""
        self.show_surface = not self.show_surface
        print(f"B-spline surface {'visible' if self.show_surface else 'hidden'}")
    
    def move_control_point(self, x, y):
        """Move the selected control point based on screen coordinates."""
        if self.selected_point is None:
            return
        
        # Get window size
        width, height = glfw.get_window_size(self.window)
        
        # Get the current point's z-depth in screen space
        point = self.patches[self.active_patch][self.selected_point]
        
        # Get viewport
        viewport = glGetIntegerv(GL_VIEWPORT)
        
        # Get the modelview and projection matrices
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Get the z-depth of the current point
        win_x, win_y, winZ = gluProject(point[0], point[1], point[2], modelview, projection, viewport)
        
        # Unproject the new position
        new_pos = unproject_screen_coords(x, y, width, height, winZ)
        
        # Update the control point position
        self.patches[self.active_patch][self.selected_point] = new_pos
        
        # Update the surface
        self.update_surface()
    
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard inputs."""
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
                self.create_new_patch()
            elif key == glfw.KEY_TAB:
                self.switch_patch(True)
            elif key == glfw.KEY_P:
                self.switch_patch(False)
    
    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button events."""
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
                width, height = glfw.get_window_size(self.window)
                self.selected_point = find_closest_control_point(
                    x, y, 
                    self.patches[self.active_patch], 
                    width, height
                )
                if self.selected_point is not None:
                    self.dragging = True
            elif action == glfw.RELEASE:
                self.dragging = False
                self.selected_point = None
    
    def cursor_pos_callback(self, window, x, y):
        """Handle mouse movement."""
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
        """Handle scrolling for zoom."""
        # Zoom in/out
        self.camera_distance -= yoffset * 0.1
        if self.camera_distance < 0.1:
            self.camera_distance = 0.1
    
    def resize_callback(self, window, width, height):
        """Handle window resize."""
        # Update viewport
        glViewport(0, 0, width, height)
    
    def render(self):
        """Render the scene."""
        # Clear the buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up camera and lighting
        width, height = glfw.get_window_size(self.window)
        setup_camera(width, height, self.camera_distance, self.camera_angle_x, self.camera_angle_y)
        
        # Draw the control grid
        if self.show_control_grid:
            draw_control_grid(self.patches[self.active_patch], self.grid_edges)
        
        # Draw the B-spline surface
        if self.show_surface:
            draw_surface(self.surface_v, self.surface_f)
    
    def run(self):
        """Run the main application loop."""
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