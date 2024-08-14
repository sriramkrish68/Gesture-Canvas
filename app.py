import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Color mappings
color_map = {
    'Red': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0),
    'Yellow': (0, 255, 255),
    'Cyan': (255, 255, 0),
    'Magenta': (255, 0, 255)
}
color_index = 'Red'

# Create a blank canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
eraser_size = 50
drawing = False
prev_x, prev_y = None, None

def is_index_finger_open(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    return index_tip.y < index_dip.y

def select_tool():
    with open('selection.txt', 'r') as file:
        selected_color, selected_tool = file.read().splitlines()
    return selected_color, selected_tool

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    selected_color, selected_tool = select_tool()
    selected_color = color_map[selected_color]

    # Process hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = img.shape
            index_finger_tip_x = int(index_finger_tip.x * w)
            index_finger_tip_y = int(index_finger_tip.y * h)

            if selected_tool == 'Clear All':
                canvas.fill(0)
                continue

            if is_index_finger_open(hand_landmarks) and index_finger_tip_y > 50:
                if drawing:
                    if selected_tool == 'Eraser':
                        cv2.circle(canvas, (index_finger_tip_x, index_finger_tip_y), eraser_size, (0, 0, 0), -1)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (index_finger_tip_x, index_finger_tip_y), selected_color, 5)
                prev_x, prev_y = index_finger_tip_x, index_finger_tip_y
                drawing = True
            else:
                drawing = False
                prev_x, prev_y = None, None

            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    combined_frame = np.hstack((img, canvas))
    cv2.imshow("Air Canvas", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
