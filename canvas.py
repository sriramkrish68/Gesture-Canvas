import numpy as np
import cv2

class Canvas:
    def __init__(self):
        self.image = np.zeros((720, 1280, 3), np.uint8)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        self.current_color = self.colors[0]

    def draw(self, position, color):
        cv2.circle(self.image, position, 5, color, -1)
