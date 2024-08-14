import cv2
import numpy as np

def draw_color_palette(frame, colors):
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (i*60, 0), ((i+1)*60, 60), color, -1)

def combine_images(img1, img2):
    return np.hstack((img1, img2))
