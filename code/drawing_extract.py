import matplotlib.pyplot as plt
import numpy as np

# get the extremities so as to start at the correct place
# careful for the drawing : extremities must contain only one neighbor
def find_feature_ext(father,i,j,nb_features,union,res_dr,points_dr,dessin):
    # according to a point of coordinate (i,j) we will look whether its 8 neighbors are in the same feature
    grad_x = [-1,0,1]
    grad_y = [-1,0,1]
    n,m = union.shape
    for e1 in grad_x:
        for e2 in grad_y:
            x=i+e1
            y=j+e2
            if m*x+y!=father and dessin[x,y,0]!=255. and union[x,y]==m*x+y:    
                union[x,y]=father
                points_dr[father][x,y] = get_neighbors(x,y,dessin)
                find_feature_ext(father,x,y,nb_features,union,res_dr,points_dr,dessin)
                

def get_extremities(dessin) :
    # given the drawing, will return the number of neighbors for each point that is not white
    
    # the points of the drawing
    points_dr = {}    
    
    # union find for all the features
    union=np.zeros(dessin.shape[:2])
    (n,m)=union.shape
    for i in range(n):
        for j in range(m):
            union[i,j]=m*i+j
            

    res_dr = {} # temporary contains the points, that still need labeling before going into res
    nb_features=0
    for j in range(m):
        for i in range(n):
            if dessin[i,j,0]==255.:
                union[i,j]=0
            #premier pixel d'un feature
            elif union[i,j]==m*i+j:
                points_dr[m*i+j] = {}
                points_dr[m*i+j][i,j] = get_neighbors(i,j,dessin)
                
                find_feature_ext(m*i+j,i,j,nb_features,union,res_dr,points_dr,dessin)
                res_dr[str(nb_features)] = points_dr[m*i+j]
                nb_features+=1
    return res_dr

def get_neighbors(i,j,dessin) :
    # numbers of neighbors of the (i,j) point
    S = -1
    neighbors = []
    for e1 in [-1,0,1] :
        for e2 in [-1,0,1] : 
            if dessin[i+e1,j+e2,0] != 255 :
                S +=1
                neighbors.append([i+e1,j+e2])
    
    return S


def find_feature(father,i,j,nb_features,union,res_draw,res_dr,points_dr,dessin):
    # according to a point we know is on the feature identified as *father*, we check if its neighbors are also.
    grad_x = [-1,0,1]
    grad_y = [-1,0,1]
    n,m = union.shape
    for e1 in grad_x:
        for e2 in grad_y:
            x=i+e1
            y=j+e2
            if m*x+y!=father and dessin[x,y,0]!=255. and union[x,y]==m*x+y:    
                union[x,y]=father
                points_dr[father].append([y,x])
                res_draw[str(nb_features)][x,y] = 255
                find_feature(father,x,y,nb_features,union,res_draw,res_dr,points_dr,dessin)
                

def compute_init_drawing(dessin,ext) :
    # the points of the drawing
    points_dr = {}    
    
    # union find for all the features
    union=np.zeros(dessin.shape[:2])
    (n,m)=union.shape
    for i in range(n):
        for j in range(m):
            union[i,j]=m*i+j
            
    res = {} # final result, after getting the right label
    res_draw = {} # contains one image per feature, to help labeling
    res_dr = {} # temporary contains the points, that still need labeling before going into res
    nb_features = 0
    
    # we will look for each feature that we found previously
    for x,y in ext :
        points_dr[m*x+y] = [[y,x]]
        res_draw[str(nb_features)] = np.zeros(union.shape)
        res_draw[str(nb_features)][x,y] = 255
        find_feature(m*x+y,x,y,nb_features,union,res_draw,res_dr,points_dr,dessin)
        res_dr[str(nb_features)] = points_dr[m*x+y]
        nb_features+=1
        
    # asking which part it is            
    for k in res_draw :
        plt.imshow(res_draw[k])
        plt.show()
        feature = input("Which feature is it ?")
        res[feature] = res_dr[k]
        
    return res