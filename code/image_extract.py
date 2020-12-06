import dlib
import cv2
from imutils import face_utils
import numpy as np

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def compute_init_pict(image) :
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    # load the input image, resize it, and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    res = {}
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            res[name] = {}
            res[name]["points"] = []
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            for (indice,(x, y)) in enumerate(shape[i:j]):
                res[name]["points"].append(shape[i:j][indice])

    mat_res = np.zeros(image.shape[:2])

    return res

def get_bottom(L) :
    # return the bottomest point
    bot = L[0][1]
    for k in L :
        if k[1]>bot :
            bot = k[1]
    return bot

def add_hair(res_im) :
    # coordinates of the face
    mid = res_im["jaw"]["points"][0][1]
    left = res_im["jaw"]["points"][0][0]
    right = res_im["jaw"]["points"][-1][0]
    left1 = res_im["right_eye"]["points"][0][0]
    right1 = res_im["left_eye"]["points"][3][0]
    bottom = get_bottom(res_im["jaw"]["points"])
    middle_mouth = res_im["mouth"]["points"][3][0]
    dist = bottom - mid
    
    # points
    top_left = [left,mid-dist]
    top_right = [right,mid-dist]
    hair_left = [left1,int(mid-0.8*dist)]
    hair_right = [right1,int(mid-0.8*dist)]
    top_hair = [middle_mouth,int(mid-1.3*dist)]
    bottom_hair = [middle_mouth,int(mid-0.8*dist)]
    hair = [top_left,top_hair,top_right,hair_right,bottom_hair,hair_left]
    
    res_im["hair"] = {}
    res_im["hair"]["points"] = hair
    return res_im