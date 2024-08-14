


# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize Mediapipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils

# # Colors for drawing
# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
# color_index = 0
# eraser_size = 50

# # Create a blank canvas
# canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# # Setup the camera feed
# cap = cv2.VideoCapture(0)

# # Variables for drawing
# drawing = False
# prev_x, prev_y = None, None
# eraser_mode = False

# def select_tool(index_finger_tip_x, index_finger_tip_y):
#     global color_index, eraser_mode, canvas
#     if index_finger_tip_y < 50:  # Check if the finger is in the palette area
#         for i in range(len(colors)):
#             if 60 * i < index_finger_tip_x < 60 * (i + 1):
#                 color_index = i
#                 eraser_mode = False
#                 return colors[i]
#         # Check if the eraser is selected
#         if 60 * len(colors) < index_finger_tip_x < 60 * (len(colors) + 1):
#             eraser_mode = True
#             return (0, 0, 0)
#         # Check if the "Clear All" button is selected
#         if 60 * (len(colors) + 1) < index_finger_tip_x < 60 * (len(colors) + 2):
#             canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
#             return colors[color_index]
#     return colors[color_index]

# def is_index_finger_open(hand_landmarks):
#     index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
#     return index_tip.y < index_dip.y

# def are_all_fingers_open(hand_landmarks):
#     fingers_open = []
#     for finger_tip, finger_dip in [(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP),
#                                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
#                                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP),
#                                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP),
#                                    (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP)]:
#         finger_tip_y = hand_landmarks.landmark[finger_tip].y
#         finger_dip_y = hand_landmarks.landmark[finger_dip].y
#         fingers_open.append(finger_tip_y < finger_dip_y)
#     return all(fingers_open)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally for natural interaction
#     frame = cv2.flip(frame, 1)

#     # Convert the frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame and detect hand landmarks
#     result = hands.process(frame_rgb)

#     # Draw the color selection palette, eraser, and "Clear All" button
#     for i, color in enumerate(colors):
#         if i == color_index:
#             cv2.rectangle(frame, (60 * i, 0), (60 * (i + 1), 50), (255, 255, 255), -1)
#             cv2.rectangle(frame, (60 * i + 3, 3), (60 * (i + 1) - 3, 47), color, -1)
#         else:
#             cv2.rectangle(frame, (60 * i, 0), (60 * (i + 1), 50), color, -1)

#     cv2.rectangle(frame, (60 * len(colors), 0), (60 * (len(colors) + 1), 50), (255, 255, 255), 2)
#     cv2.putText(frame, "Eraser", (60 * len(colors) + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     cv2.rectangle(frame, (60 * (len(colors) + 1), 0), (60 * (len(colors) + 2), 50), (255, 255, 255), 2)
#     cv2.putText(frame, "Clear All", (60 * (len(colors) + 1) + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     # If hand landmarks are detected
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             # Get the index finger tip position
#             index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             h, w, _ = frame.shape
#             index_finger_tip_x = int(index_finger_tip.x * w)
#             index_finger_tip_y = int(index_finger_tip.y * h)

#             # Select color, eraser, or "Clear All"
#             selected_color = select_tool(index_finger_tip_x, index_finger_tip_y)

#             # Draw a cursor at the fingertip position
#             if drawing or eraser_mode:
#                 cv2.circle(frame, (index_finger_tip_x, index_finger_tip_y), 10, (255, 255, 255), 2)

#             # Check if all fingers are open for erasing
#             if are_all_fingers_open(hand_landmarks):
#                 cv2.circle(canvas, (index_finger_tip_x, index_finger_tip_y), eraser_size, (0, 0, 0), -1)
#             # Check if the index finger is open for drawing
#             elif is_index_finger_open(hand_landmarks) and index_finger_tip_y > 50:
#                 if drawing:
#                     if eraser_mode:
#                         cv2.circle(canvas, (index_finger_tip_x, index_finger_tip_y), eraser_size, (0, 0, 0), -1)
#                     else:
#                         cv2.line(canvas, (prev_x, prev_y), (index_finger_tip_x, index_finger_tip_y), selected_color, 5)
#                 prev_x, prev_y = index_finger_tip_x, index_finger_tip_y
#                 drawing = True
#             else:
#                 drawing = False
#                 prev_x, prev_y = None, None

#             # Draw hand landmarks and connections
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # Show the frames
#     cv2.imshow("Air Canvas - Feed", frame)
#     cv2.imshow("Air Canvas - Drawing", canvas)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Colors for drawing
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
color_index = 0
eraser_size = 50
palette_gap = 10

# Create a blank canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Setup the camera feed
cap = cv2.VideoCapture(0)

# Variables for drawing
drawing = {0: False, 1: False}  # Keep track of drawing state for both hands
prev_pos = {0: (None, None), 1: (None, None)}  # Keep track of previous positions for both hands
eraser_mode = {0: False, 1: False}  # Keep track of eraser mode for both hands

def select_tool(hand_index, index_finger_tip_x, index_finger_tip_y):
    global color_index, eraser_mode, canvas
    if index_finger_tip_y < 50:  # Check if the finger is in the palette area
        for i in range(len(colors)):
            if (60 + palette_gap) * i < index_finger_tip_x < (60 + palette_gap) * i + 60:
                color_index = i
                eraser_mode[hand_index] = False
                return colors[i]
        # Check if the eraser is selected
        if (60 + palette_gap) * len(colors) < index_finger_tip_x < (60 + palette_gap) * len(colors) + 60:
            eraser_mode[hand_index] = True
            return (0, 0, 0)
        # Check if the "Clear All" button is selected
        if (60 + palette_gap) * (len(colors) + 1) < index_finger_tip_x < (60 + palette_gap) * (len(colors) + 1) + 60:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            return colors[color_index]
    return colors[color_index]

def is_index_finger_open(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    return index_tip.y < index_dip.y

def are_all_fingers_open(hand_landmarks):
    fingers_open = []
    for finger_tip, finger_dip in [(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP),
                                   (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
                                   (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP),
                                   (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP),
                                   (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP)]:
        finger_tip_y = hand_landmarks.landmark[finger_tip].y
        finger_dip_y = hand_landmarks.landmark[finger_dip].y
        fingers_open.append(finger_tip_y < finger_dip_y)
    return all(fingers_open)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hand landmarks
    result = hands.process(frame_rgb)

    # Draw the color selection palette, eraser, and "Clear All" button with gaps
    for i, color in enumerate(colors):
        if i == color_index:
            cv2.rectangle(frame, ((60 + palette_gap) * i, 0), ((60 + palette_gap) * i + 60, 50), (255, 255, 255), -1)
            cv2.rectangle(frame, ((60 + palette_gap) * i + 3, 3), ((60 + palette_gap) * i + 57, 47), color, -1)
        else:
            cv2.rectangle(frame, ((60 + palette_gap) * i, 0), ((60 + palette_gap) * i + 60, 50), color, -1)

    cv2.rectangle(frame, ((60 + palette_gap) * len(colors), 0), ((60 + palette_gap) * len(colors) + 60, 50), (255, 255, 255), 2)
    cv2.putText(frame, "Eraser", ((60 + palette_gap) * len(colors) + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(frame, ((60 + palette_gap) * (len(colors) + 1), 0), ((60 + palette_gap) * (len(colors) + 1) + 60, 50), (255, 255, 255), 2)
    cv2.putText(frame, "Clear All", ((60 + palette_gap) * (len(colors) + 1) + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # If hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Get the index finger tip position
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            index_finger_tip_x = int(index_finger_tip.x * w)
            index_finger_tip_y = int(index_finger_tip.y * h)

            # Select color, eraser, or "Clear All"
            selected_color = select_tool(hand_index, index_finger_tip_x, index_finger_tip_y)

            # Draw a cursor at the fingertip position
            if drawing[hand_index] or eraser_mode[hand_index]:
                cv2.circle(frame, (index_finger_tip_x, index_finger_tip_y), 10, (255, 255, 255), 2)

            # Check if all fingers are open for erasing
            if are_all_fingers_open(hand_landmarks):
                cv2.circle(canvas, (index_finger_tip_x, index_finger_tip_y), eraser_size, (0, 0, 0), -1)
            # Check if the index finger is open for drawing
            elif is_index_finger_open(hand_landmarks) and index_finger_tip_y > 50:
                if drawing[hand_index]:
                    if eraser_mode[hand_index]:
                        cv2.circle(canvas, (index_finger_tip_x, index_finger_tip_y), eraser_size, (0, 0, 0), -1)
                    else:
                        prev_x, prev_y = prev_pos[hand_index]
                        if prev_x is not None and prev_y is not None:
                            cv2.line(canvas, (prev_x, prev_y), (index_finger_tip_x, index_finger_tip_y), selected_color, 5)
                prev_pos[hand_index] = (index_finger_tip_x, index_finger_tip_y)
                drawing[hand_index] = True
            else:
                drawing[hand_index] = False
                prev_pos[hand_index] = (None, None)

            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frames
    cv2.imshow("Air Canvas - Feed", frame)
    cv2.imshow("Air Canvas - Drawing", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


