import numpy as np
from numpy.linalg import inv

RIGHT = "RIGHT"
LEFT = "LEFT"

def get_min_max(x,y) :
    if x>y : 
        return y,x
    return x,y

# whether the point in inside the polygone, here triangle
def inside_convex_polygon(point, vertices):
    # inside the triangle
    previous_side = None
    n_vertices = len(vertices)
    for n in range(n_vertices):
        a, b = vertices[n], vertices[(n+1)%n_vertices]
        affine_segment = v_sub(b, a)
        affine_point = v_sub(point, a)
        current_side = get_side(affine_segment, affine_point)
        if current_side is None:
            #outside or over an edge
            # check if over an edge
            min_x,max_x = get_min_max(a[0],b[0])
            min_y,max_y = get_min_max(a[1],b[1])
            if point[0] >= min_x and point[0] <= max_x and point[1]>=min_y and point[1] <= max_y :
                return True
            return False #outside
        elif previous_side is None: #first segment
            previous_side = current_side
        elif previous_side != current_side:
            return False
    return True

def on_convexe_polygone(point,vertices) : 
    # on the vertices or edges
    for i in range(len(vertices)) : # will be 3 cause it is a triangle
        if (vertices[i][0] - vertices[(i+1)%len(vertices)][0]) == 0 and vertices[i][1]== point[1] : 
            return False
        if point[1] == (vertices[i][1] - vertices[(i+1)%len(vertices)][1]) / (vertices[i][0] - vertices[(i+1)%len(vertices)][0]) * (point[0] -vertices[i][0]) + vertices[i][1] : 
            return True # aka the point is on the edge betwen vertice[i] and vertice[i+1]
    return False

def get_side(a, b):
    x = cosine_sign(a, b)
    if x < 0:
        return LEFT
    elif x > 0: 
        return RIGHT
    else:
        return None

def v_sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def cosine_sign(a, b):
    return a[0]*b[1]-a[1]*b[0]

def get_face(point,F,V) :
    # given a point, return the face in which he is contained
    for ind,k in enumerate(F) :
        if inside_convex_polygon(point,[V[k[0]],V[k[1]],V[k[2]]]) :
            return ind 

# point = alpha * point[0] + beta * point[1] + gamma * point[2]
def get_coefs(point,vertices) :
    # one point and the three vertices
    M = np.ones((3,3))
    M[1,0] = vertices[0][0]
    M[1,1] = vertices[1][0]
    M[1,2] = vertices[2][0]
    
    M[2,0] = vertices[0][1]
    M[2,1] = vertices[1][1]
    M[2,2] = vertices[2][1]
    
    Y = np.zeros((3,1))
    Y[0,0] = 1
    Y[1,0] = point[0]
    Y[2,0] = point[1]
    try :
        return inv(M) @ Y
    except :
        # the face is flat
        # If the face is flat, which point are more important?
        default = np.zeros((3,1))
        default[0,0] = np.linalg.norm(np.array(vertices[0])-np.array(point))
        default[1,0] = np.linalg.norm(np.array(vertices[1])-np.array(point))
        default[2,0] = np.linalg.norm(np.array(vertices[2])-np.array(point))
        norm = np.linalg.norm(np.array(vertices[0])-np.array(point)) + np.linalg.norm(np.array(vertices[1])-np.array(point)) +np.linalg.norm(np.array(vertices[2])-np.array(point))
        default = default / norm
        default[0,0] = 1-default[0,0]
        default[1,0] = 1-default[1,0]
        default[2,0] = 1-default[2,0]
        default = default / np.linalg.norm(default)
        return default