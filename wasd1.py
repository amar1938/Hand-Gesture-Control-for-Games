import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Key, Controller
import time

# Set the desired width and height for the webcam screen
width, height = 540, 480  # Lower resolution for improved performance

# Open the webcam and set the width and height for capturing video frames
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Initialize the HandDetector with a confidence threshold and max no. of hands to detect (1 or 2)
detector = HandDetector(detectionCon=0.7, maxHands=1)

# Create a keyboard controller to simulate key presses
keyboard = Controller()

# Initialize the previous finger status as an invalid state
prev_fingers = None

# Variable to count the number of frames to skip for hand detection
skip_frames = 0

# Variables to track the last time each key was pressed
last_acceleration_time = 0
last_brake_time = 0
last_steer_left_time = 0
last_steer_right_time = 0

# Define the delay between key presses in seconds
key_delay = 0.5

def invert_camera(image):
    return cv2.flip(image, 1)

try:
    while True:
        # Read a frame from the webcam
        ret, img = cap.read()

        if not ret:
            break

        # Reduce the frame resolution for faster processing
        img = cv2.resize(img, (width, height))

        # Invert the camera feed
        img = invert_camera(img)

        if skip_frames == 0:
            # Detect the hand in the frame using the HandDetector
            hands, img = detector.findHands(img, draw=False)

            # Reset skip_frames to its initial value after detection
            skip_frames = 5

        # Decrement skip_frames to skip the next few frames for hand detection
        skip_frames -= 1

        # Check if a hand is detected
        if hands:
            hand = hands[0]  # Get the first (and only) detected hand

            # Draw hand landmarks and bounding box on the image
            bbox = hand["bbox"]  # Bounding box info x,y,w,h
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            for lm in hand["lmList"]:
                cv2.circle(img, (lm[0], lm[1]), 5, (0, 255, 0), cv2.FILLED)

            # Get the hand landmark points
            if "lmList" in hand:
                hand_landmarks = hand["lmList"]
                hand_np = np.array(hand_landmarks)

                # Calculate the finger tips y-coordinates
                fingertips_y = hand_np[[4, 8, 12, 16, 20]][:, 1]

                # Check which fingers are up
                fingers_up = fingertips_y < hand_landmarks[2][1]

                # Convert the finger status to a NumPy array
                fingers_np = np.array(fingers_up)

                # Check if the finger status has changed
                if prev_fingers is None or not np.array_equal(fingers_np, prev_fingers):
                    prev_fingers = fingers_np

                    current_time = time.time()

                    # Only press the keyboard when the hand gesture changes
                    if np.all(fingers_np):
                        if current_time - last_acceleration_time > key_delay:
                            # If all fingers are open, simulate pressing the W key (accelerate)
                            keyboard.press('w')
                            keyboard.release('s')
                            last_acceleration_time = current_time
                    elif not np.any(fingers_np):
                        if current_time - last_brake_time > key_delay:
                            # If all fingers are closed, simulate pressing the S key (brake/reverse)
                            keyboard.press('s')
                            keyboard.release('w')
                            last_brake_time = current_time
                    else:
                        # Release both keys if fingers are in mixed states
                        keyboard.release('w')
                        keyboard.release('s')

                    # Additional logic to handle steering
                    # Here, we'll use the x-coordinate of the index finger (landmark 8)
                    index_finger_x = hand_landmarks[8][0]
                    if index_finger_x < hand_landmarks[0][0] - 20:
                        if current_time - last_steer_left_time > key_delay:
                            # Steer left if the index finger is significantly to the left of the wrist
                            keyboard.press('a')
                            keyboard.release('d')
                            last_steer_left_time = current_time
                    elif index_finger_x > hand_landmarks[0][0] + 20:
                        if current_time - last_steer_right_time > key_delay:
                            # Steer right if the index finger is significantly to the right of the wrist
                            keyboard.press('d')
                            keyboard.release('a')
                            last_steer_right_time = current_time
                    else:
                        # Release both steering keys if hand is centered
                        keyboard.release('d')
                        keyboard.release('a')

            else:
                print("Error: 'lmList' key not found in hand dictionary")

        else:
            prev_fingers = None
            # If no hand is detected, release all control keys
            keyboard.release('w')
            keyboard.release('s')
            keyboard.release('a')
            keyboard.release('d')

        # Show the image with hand gesture information
        cv2.imshow("hand_gesture_control", img)

        # Check for the "q" key press to exit the infinite loop and close the program
        if cv2.waitKey(1) == ord("q"):
            break

except Exception as e:
    print("Exception:", e)

finally:
    # Release the webcam and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
