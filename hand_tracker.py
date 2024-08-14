import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        self.color_selected = (255, 0, 0)  # Default color

    def detect_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)
        return result.multi_hand_landmarks

    def select_color(self, frame):
        # Logic for selecting color based on finger position
        if self.detect_hands(frame):
            hand_landmarks = self.detect_hands(frame)[0]
            index_finger_tip = self.get_index_finger_tip(hand_landmarks)
            h, w, _ = frame.shape
            if index_finger_tip and index_finger_tip[1] < 60:
                self.color_selected = frame[0, index_finger_tip[0]]
        return self.color_selected

    def get_index_finger_tip(self, hand_landmarks):
        if hand_landmarks:
            x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 1280)
            y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 720)
            return x, y
        return None
