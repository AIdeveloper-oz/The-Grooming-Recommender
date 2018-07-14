# importing libraries
import cv2
import dlib
import numpy
import glob

#importing supporting modules
import get_landmarks
import get_face_mask
import correct_colours
import read_im_and_landmarks
import swappy
import transformation_from_points
import warp_im
###########################################################################################################################################
# Some parameters for merging
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#############################################################################################################################################
data = []
files = glob.glob("images/*.jpg") #Enter the directory of images folder
for myFile in files:
    print(myFile)
    image = cv2.imread(myFile)
    data.append(image)

for f in range(len(data)):
    image1 = data[f]
    image2 = cv2.imread('Animesh.jpeg') #Enter your image here!

    swapped1 = swappy(image1, image2)
    out = cv2.pyrDown(swapped1)
    cv2.imshow("Face Swap" + str(f), swapped1)

    cv2.waitKey(0)
cv2.destroyAllWindows()
