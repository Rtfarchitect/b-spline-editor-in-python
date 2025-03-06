import numpy as np

def create_uniform_knots(n, k):
    """
    Create a uniform knot vector with multiplicity k+1 at the ends.
    
    Args:
        n: Number of control points
        k: Degree of the B-spline
    
    Returns:
        Numpy array containing the knot vector
    """
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

def basis_function(u, i, k, knots):
    """
    Evaluate the B-spline basis function of degree k at parameter u.
    
    Args:
        u: Parameter value
        i: Control point index
        k: Degree of the basis function
        knots: Knot vector
    
    Returns:
        Value of the basis function
    """
    if k == 0:
        if knots[i] <= u < knots[i+1] or (abs(u - knots[i+1]) < 1e-10 and abs(knots[i+1] - knots[-1]) < 1e-10):
            return 1.0
        else:
            return 0.0
    
    d1 = knots[i+k] - knots[i]
    d2 = knots[i+k+1] - knots[i+1]
    
    f1 = 0.0 if d1 < 1e-10 else (u - knots[i]) / d1
    f2 = 0.0 if d2 < 1e-10 else (knots[i+k+1] - u) / d2
    
    return f1 * basis_function(u, i, k-1, knots) + f2 * basis_function(u, i+1, k-1, knots)

def evaluate_surface_point(u, v, control_pts, nu, nv, degree, u_knots, v_knots):
    """
    Evaluate surface point at parameter (u, v).
    
    Args:
        u, v: Parameter values
        control_pts: Control points array
        nu, nv: Number of control points in u and v directions
        degree: Degree of the B-spline
        u_knots, v_knots: Knot vectors
    
    Returns:
        3D point on the surface
    """
    point = np.zeros(3)
    
    control_pts_reshaped = control_pts.reshape(nu, nv, 3)
    
    for i in range(nu):
        for j in range(nv):
            basis_u = basis_function(u, i, degree, u_knots)
            basis_v = basis_function(v, j, degree, v_knots)
            point += control_pts_reshaped[i, j] * basis_u * basis_v
    
    return point

def generate_surface_mesh(control_pts, nu, nv, degree, u_knots, v_knots, resolution=20):
    """
    Generate a surface mesh for visualization.
    
    Args:
        control_pts: Control points array
        nu, nv: Number of control points in u and v directions
        degree: Degree of the B-spline
        u_knots, v_knots: Knot vectors
        resolution: Number of points in each direction
    
    Returns:
        vertices and faces for the surface mesh
    """
    vertices = []
    faces = []
    
    u_steps = resolution
    v_steps = resolution
    
    # Create vertices
    for i in range(u_steps + 1):
        u = i / u_steps
        for j in range(v_steps + 1):
            v = j / v_steps
            vertices.append(evaluate_surface_point(u, v, control_pts, nu, nv, degree, u_knots, v_knots))
    
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

def create_control_grid(nu, nv):
    """
    Create initial control points and grid connectivity.
    
    Args:
        nu, nv: Number of control points in u and v directions
    
    Returns:
        control_points, grid_edges
    """
    # Create a grid of control points
    u = np.linspace(-1, 1, nu)
    v = np.linspace(-1, 1, nv)
    
    control_points = np.zeros((nu * nv, 3))
    
    idx = 0
    for i in range(nu):
        for j in range(nv):
            control_points[idx] = [u[i], v[j], 0.0]
            idx += 1
    
    # Create a connectivity list for the control grid
    grid_edges = []
    for i in range(nu):
        for j in range(nv-1):
            grid_edges.append([i*nv + j, i*nv + j + 1])
    
    for j in range(nv):
        for i in range(nu-1):
            grid_edges.append([i*nv + j, (i+1)*nv + j])
    
    return control_points, np.array(grid_edges)
