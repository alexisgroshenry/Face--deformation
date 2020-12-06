# Face deformation based on childish drawing

Our project consists in, given two inputs (a *photo of the face of a person* and a *childish drawing of a face*), to produce the same face, but distorted according to the drawing.

## Requirement

You need to install [dlib](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) via the link for information.

For cv2, matplitlib.pyplot, spicy and numpy, just type this in the command prompt.

```bash
pip install opencv
pip install matplotlib
pip install spicy
pip install numpy
```

You also need to download the model [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract it in the *model* directory.

## Usage

### First time

Use the jupyter notebook. Every action is explained, and you go step by step through the process. If you want to use your own image, just use this code in the **Drawing and image** part.

```python
# init the drawing and picture
dessin = cv2.imread('drawings/your_drawing.png') 
image = cv2.imread("images/your_image.png")
# resize
image = cv2.resize(image,dessin.shape[:2][::-1])
```

### Other times

You can also use *deform.py* via the commande prompt. Simply paste this code. The output image will be in result named *your_image-your_drawing.png*.

```bash
python deform.py -i your_image.png -d your_drawing.png
```
