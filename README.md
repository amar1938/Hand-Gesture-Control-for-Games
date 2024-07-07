Project Description
This project is a hand gesture control system designed to simulate keyboard key presses using a webcam for input. The system leverages the cvzone and opencv-python libraries to capture and process video frames, detecting hand gestures in real time. Depending on the recognized gestures, the system simulates key presses to control acceleration, braking, and steering in a driving game or other applications. The pynput library is used to handle the simulation of keyboard events.

Key Features
Hand Detection: Utilizes the cvzone.HandTrackingModule to detect and track hand gestures.
Gesture Recognition: Identifies specific hand gestures, such as all fingers up, all fingers down, and index finger pointing left or right, to trigger corresponding keyboard actions.
Keyboard Control: Simulates key presses (e.g., 'w' for acceleration, 's' for braking, 'a' for steering left, and 'd' for steering right) using the pynput library.
Performance Optimization: Reduces frame resolution and skips frames to improve processing speed.
Visual Feedback: Displays the webcam feed with annotated hand landmarks and bounding boxes for real-time feedback.
Required Packages
To run this project, the following Python packages need to be installed:

OpenCV: For capturing and processing video frames.
NumPy: For handling numerical operations and array manipulation.
cvzone: For hand tracking and gesture detection.
pynput: For simulating keyboard key presses.
