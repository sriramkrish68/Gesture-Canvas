import cv2
from hand_tracker import HandTracker
from canvas import Canvas
from utils import draw_color_palette, combine_images

def main():
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker()
    canvas = Canvas()

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Resize the frame to match the canvas height
        frame_resized = cv2.resize(frame, (canvas.image.shape[1], canvas.image.shape[0]))

        hand_landmarks = hand_tracker.detect_hands(frame_resized)
        selected_color = hand_tracker.select_color(frame_resized)
        
        if hand_landmarks:
            index_finger_tip = hand_tracker.get_index_finger_tip(hand_landmarks)
            if index_finger_tip:
                canvas.draw(index_finger_tip, selected_color)

        draw_color_palette(frame_resized, canvas.colors)
        combined_frame = combine_images(frame_resized, canvas.image)
        
        cv2.imshow("Air Canvas", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
