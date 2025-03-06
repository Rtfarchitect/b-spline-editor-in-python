import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

def setup_lighting():
    """Setup basic lighting for the scene."""
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    
    light_position = [1.0, 1.0, 1.0, 0.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

def setup_camera(width, height, camera_distance, camera_angle_x, camera_angle_y):
    """Setup camera projection and position."""
    # Set up projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = width / height
    gluPerspective(45.0, aspect, 0.1, 100.0)
    
    # Set up modelview matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Set up camera
    gluLookAt(0, 0, camera_distance,
             0, 0, 0,
             0, 1, 0)
    
    # Apply camera rotations
    glRotatef(camera_angle_x * 180.0 / np.pi, 1, 0, 0)
    glRotatef(camera_angle_y * 180.0 / np.pi, 0, 1, 0)

def draw_control_grid(control_points, grid_edges):
    """Draw the control points and grid lines."""
    glDisable(GL_LIGHTING)
    
    # Draw control points
    glPointSize(5.0)
    glBegin(GL_POINTS)
    glColor3f(0.0, 0.0, 1.0)  # Blue
    for point in control_points:
        glVertex3f(point[0], point[1], point[2])
    glEnd()
    
    # Draw grid lines
    glColor3f(1.0, 0.0, 0.0)  # Red
    glBegin(GL_LINES)
    for edge in grid_edges:
        p1 = control_points[edge[0]]
        p2 = control_points[edge[1]]
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
    glEnd()
    
    glEnable(GL_LIGHTING)

def draw_surface(vertices, faces):
    """Draw the B-spline surface with lighting."""
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
    for face in faces:
        # Calculate face normal
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        
        glNormal3f(normal[0], normal[1], normal[2])
        glVertex3f(v0[0], v0[1], v0[2])
        glVertex3f(v1[0], v1[1], v1[2])
        glVertex3f(v2[0], v2[1], v2[2])
    glEnd()

def unproject_screen_coords(x, y, window_width, window_height, z_depth=None):
    """
    Convert screen coordinates to 3D world coordinates.
    
    Args:
        x, y: Screen coordinates
        window_width, window_height: Window dimensions
        z_depth: Optional depth value (if None, returns ray direction)
    
    Returns:
        3D point or ray (near and far points)
    """
    # Get the viewport
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    # Get the modelview and projection matrices
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    
    # Convert to NDC
    y = window_height - y  # Invert Y since OpenGL has origin at bottom-left
    
    if z_depth is None:
        # Return ray (near and far points)
        nearPos = gluUnProject(x, y, 0.0, modelview, projection, viewport)
        farPos = gluUnProject(x, y, 1.0, modelview, projection, viewport)
        return np.array(nearPos), np.array(farPos)
    else:
        # Return specific point at given depth
        return np.array(gluUnProject(x, y, z_depth, modelview, projection, viewport))

def find_closest_control_point(x, y, control_points, window_width, window_height, threshold=0.2):
    """
    Find the closest control point to the screen coordinates.
    
    Args:
        x, y: Screen coordinates
        control_points: Array of control points
        window_width, window_height: Window dimensions
        threshold: Maximum allowed distance
    
    Returns:
        Index of closest control point or None if none is close enough
    """
    # Get ray from screen coordinates
    nearPos, farPos = unproject_screen_coords(x, y, window_width, window_height)
    
    # Calculate ray direction
    ray_dir = farPos - nearPos
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    
    # Find closest control point to the ray
    min_dist = float('inf')
    closest_idx = -1
    
    for i, point in enumerate(control_points):
        # Calculate the distance from the point to the ray
        v = point - nearPos
        dist = np.linalg.norm(np.cross(v, ray_dir))
        
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    # Only select if it's reasonably close
    if min_dist < threshold:
        return closest_idx
    
    return None
