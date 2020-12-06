# import the necessary packages
import argparse
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from math import *
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.spatial import Delaunay
from scipy import ndimage
from scipy.interpolate import interp1d
import sys
sys.path.insert(1, './code')

from drawing_extract import *
from image_extract import *
from face_belong import *

parser = argparse.ArgumentParser()
parser.add_argument("-i","--image", help="image we want to deform",
                    type=str)
parser.add_argument("-d","--drawing", help="drawing we want to deform",
                    type=str)
args = parser.parse_args()

def deformation() :
        # init the drawing and picture
    print("Loading image and drawing")
    dessin = cv2.imread('drawings/{}'.format(args.drawing)) 
    image = cv2.imread("images/{}".format(args.image))
    # resize
    image = cv2.resize(image,dessin.shape[:2][::-1])

    print("Get the extremities of drawing")
    res = get_extremities(dessin)
    ext = []
    for k in res : 
        current_ext = []
        for key,values in res[k].items() :
            if values == 1 :
                current_ext.append(key)
        # how to choose? closest to (0,0)
        norm0 = np.linalg.norm(np.array(current_ext[0]))
        norm1 = np.linalg.norm(np.array(current_ext[1]))
        if norm0 > norm1 :
            ext.append(current_ext[1])
        else : 
            ext.append(current_ext[0]) 

    print("Labelling the drawing")
    res_d = compute_init_drawing(dessin,ext)

    print("Extracting image")
    res_im = compute_init_pict(image)
    res_im = add_hair(res_im)

    print("Interpolation") 
    points_tot = {}
    for k in res_d :
        points_tot[k] = {}
        points_tot[k]["drawing"] = res_d[k]
        if k == 'mouth' :
            points_tot[k]["image"] = add_points(res_im[k]["points"][:12],part=k)
        else :
            points_tot[k]["image"] = add_points(res_im[k]["points"],part=k)

    print("Same amount of points")
    for k in points_tot : 
        points_tot[k]["drawing"] = good_points(points_tot[k]["drawing"],points_tot[k]["image"])

    print("Stack points")
    image_points = []
    drawing_points = []
    for k in points_tot :
        for i in points_tot[k]["image"] :
            image_points.append(i)
        for j in points_tot[k]["drawing"] :
            drawing_points.append(j)
    
    print("Delaunay triangulation")
    points =np.array([list(coords) for coords in image_points])
    drawing_points = np.array([list(coords) for coords in drawing_points])

    # compute Delaunay method
    tri = Delaunay(points)

    print("Compute final deformation") 
    #result matrix
    deformed = np.zeros(dessin.shape)

    for i in range(dessin.shape[1]):
        if i%50 == 0 :
            print("Column {} over {}".format(i,dessin.shape[1]))
        for j in range(dessin.shape[0]):
            for f in tri.simplices:
                l=list(drawing_points[v] for v in list(f))
                l2 = list(points[v] for v in list(f))
                if inside_convex_polygon([i,j],l):
                    coefs=get_coefs([i,j],l)
                    x = int(sum(coefs[i]*l2[i][0] for i in range(3) ))
                    y = int(sum(coefs[i]*l2[i][1] for i in range(3) ))
                    # a cause des arrondis, sometimes ca dépasse
                    if x > dessin.shape[1]-1 :
                        x = dessin.shape[1] -1
                    if y > dessin.shape[0]-1 :
                        y = dessin.shape[0] - 1
                    deformed[j,i]=image[y,x]
                    break
    deformed = deformed.astype("uint8")

    print("Saving image")
    name = args.image.split(".")[0] + "-" + args.drawing.split(".")[0]
    cv2.imwrite("result/{}.png".format(name),deformed)

    return None


# add points according to cubic interpolation
# not so good for now
def add_points(points,part="jaw") :
    if part == "hair" or part == "left_eybrow" or part == "right_eyebrow" :
        return points
    x = []
    y = []
    for a,b in points:
        x.append(a)
        y.append(b)
    if not (part in ["jaw",'left_eyebrow','right_eyebrow']) :
        x.append(points[0][0])
        y.append(points[0][1])
    t = [k for k in range(len(x))]
    new_points = []
    fx = interp1d(t, x, kind='cubic')
    fy = interp1d(t, y, kind='cubic')
    for i in t : 
        try : 
            # last point in the first point if we circle
            if i != len(t)-1 and not(part in ["jaw",'left_eyebrow','right_eyebrow']) :
                x_x = fx(i)
                y_y = fy(i)
                new_points.append([int(x_x),int(y_y)])
            elif (part in ["jaw",'left_eyebrow','right_eyebrow']) : 
                x_x = fx(i)
                y_y = fy(i)
                new_points.append([int(x_x),int(y_y)])
            x_x = fx(i+0.5)
            y_y = fy(i+0.5)
            new_points.append([int(x_x),int(y_y)])
        except :
            a = 0 # actually not a mistake, but I did not know what to say
    return new_points

def good_points(origine,goal) :
    # we have origine points (greater number) and want to obtain as much points as in goal
    n = len(goal)
    count = len(origine)
    obj = []
    interval = count//n
    for i in range(n) :
        obj.append(origine[i*interval])
    return obj


    


deformation()
