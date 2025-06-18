#!/usr/bin/env python3
"""
Hand Gesture Recognition using OpenCV and MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import math


class HandGestureRecognition:
    """Hand gesture recognition using MediaPipe and OpenCV"""

    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Gesture definitions
        self.gestures = {
            "unknown": "Unknown Gesture",
            "palm": "Open Palm",
            "fist": "Fist",
            "thumbs_up": "Thumbs Up",
            "peace": "Peace Sign",
            "pointing": "Pointing",
        }

    def calculate_finger_angles(self, hand_landmarks):
        """Calculate angles between finger joints to determine finger state (open/closed)"""
        # Get coordinates of key landmarks
        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
        thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
        index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
        middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y])
        ring_tip = np.array([hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y])
        pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y])
        
        # Get middle points of each finger (for angle calculation)
        thumb_mcp = np.array([hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y])
        index_mcp = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y])
        middle_mcp = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y])
        ring_mcp = np.array([hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y])
        pinky_mcp = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y])
        
        # Calculate distances to determine if fingers are extended
        finger_states = {
            "thumb": np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_mcp - wrist),
            "index": np.linalg.norm(index_tip - wrist) > np.linalg.norm(index_mcp - wrist),
            "middle": np.linalg.norm(middle_tip - wrist) > np.linalg.norm(middle_mcp - wrist),
            "ring": np.linalg.norm(ring_tip - wrist) > np.linalg.norm(ring_mcp - wrist),
            "pinky": np.linalg.norm(pinky_tip - wrist) > np.linalg.norm(pinky_mcp - wrist),
        }
        
        return finger_states

    def recognize_gesture(self, hand_landmarks):
        """Recognize hand gesture based on finger positions"""
        if not hand_landmarks:
            return "unknown"
        
        # Get finger states (extended or not)
        finger_states = self.calculate_finger_angles(hand_landmarks)
        
        # Recognize gestures based on finger states
        if all(finger_states.values()):
            return "palm"  # All fingers extended
        
        elif not any([finger_states["index"], finger_states["middle"], 
                     finger_states["ring"], finger_states["pinky"]]):
            return "fist"  # All fingers closed
        
        elif finger_states["thumb"] and not any([finger_states["index"], finger_states["middle"], 
                                               finger_states["ring"], finger_states["pinky"]]):
            return "thumbs_up"  # Only thumb extended
        
        elif finger_states["index"] and finger_states["middle"] and not finger_states["ring"] and not finger_states["pinky"]:
            return "peace"  # Index and middle extended, others closed
        
        elif finger_states["index"] and not finger_states["middle"] and not finger_states["ring"] and not finger_states["pinky"]:
            return "pointing"  # Only index finger extended
        
        return "unknown"

    def process_frame(self, frame):
        """Process a single frame to detect and recognize hand gestures"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks and recognize gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
                
                # Recognize gesture
                gesture = self.recognize_gesture(hand_landmarks)
                
                # Get hand position for text placement
                h, w, _ = frame.shape
                x = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
                y = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h) - 20
                
                # Draw gesture name
                cv2.putText(
                    frame,
                    self.gestures[gesture],
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        
        return frame

    def run(self):
        """Run the hand gesture recognition application"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Hand Gesture Recognition started. Press 'q' to quit.")
        
        while cap.isOpened():
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Mirror the frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow("Hand Gesture Recognition", processed_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandGestureRecognition()
    app.run()
