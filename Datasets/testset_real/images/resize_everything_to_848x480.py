import cv2
import os

img_width = 848
img_height = 480

current_path = os.path.dirname(os.path.realpath(__file__))

img_list = [file for file in os.listdir(current_path) if file.endswith('.png')]

for img_name in img_list:
    img = cv2.imread(img_name)
    dim = (img_width, img_height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    cv2.imwrite(img_name, resized)