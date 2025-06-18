#!/usr/bin/env python3
"""
Vision Recognition App - Combines hand gesture recognition and face detection
using OpenCV and MediaPipe libraries.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import pickle
from PIL import Image


class VisionRecognitionApp:
    """
    A class that combines hand gesture recognition and face detection
    using OpenCV and MediaPipe libraries.
    """

    def __init__(self):
        """
        Initialize the Vision Recognition App.
        """
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
        
        # Initialize face detection
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Face recognition data
        self.known_faces = []
        self.known_face_names = []
        self.face_data_file = "face_data.pkl"
        self.face_model_file = "face_model.yml"
        self.load_face_data()
        
        # Face naming mode
        self.naming_mode = False
        self.current_face = None
        
        # Process every other frame to improve performance
        self.process_this_frame = True
        


    def calculate_finger_angles(self, hand_landmarks):
        """
        Calculate angles between finger joints to determine finger state (open/closed).
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            dict: Dictionary of finger states (True if extended, False if closed)
        """
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
        """
        Recognize hand gesture based on finger positions.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            str: Recognized gesture key
        """
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

    def process_hands(self, frame, rgb_frame):
        """
        Process hand landmarks and recognize gestures.
        
        Args:
            frame: Original BGR frame for drawing
            rgb_frame: RGB frame for MediaPipe processing
            
        Returns:
            frame: Frame with hand landmarks and gestures drawn
        """
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
    
    def load_face_data(self):
        """
        Load saved face data from file if it exists.
        """
        # Load face names
        if os.path.exists(self.face_data_file):
            try:
                with open(self.face_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', [])
                    self.known_face_names = data.get('names', [])
                print(f"Loaded {len(self.known_face_names)} face(s): {', '.join(self.known_face_names)}")
                
                # Load face recognizer model if it exists
                if os.path.exists(self.face_model_file) and self.known_faces:
                    self.face_recognizer.read(self.face_model_file)
                    print("Loaded face recognition model")
            except Exception as e:
                print(f"Error loading face data: {e}")
                self.known_faces = []
                self.known_face_names = []
    
    def save_face_data(self):
        """
        Save face data to file.
        """
        # Save face data
        data = {
            'faces': self.known_faces,
            'names': self.known_face_names
        }
        try:
            with open(self.face_data_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Save face recognizer model if we have faces
            if self.known_faces:
                self.face_recognizer.write(self.face_model_file)
            
            print(f"Saved {len(self.known_face_names)} face(s)")
        except Exception as e:
            print(f"Error saving face data: {e}")
    
    def add_face(self, name):
        """
        Add the current face with the given name.
        
        Args:
            name: Name to associate with the face
        """
        if self.current_face is not None:
            # Add to known faces
            self.known_faces.append(self.current_face)
            self.known_face_names.append(name)
            
            # Train recognizer with all faces
            faces = np.array(self.known_faces)
            labels = np.array([i for i in range(len(self.known_face_names))])
            self.face_recognizer.train(faces, labels)
            
            # Save the data
            self.save_face_data()
            print(f"Added face: {name}")
            return True
        else:
            print("No face detected to add")
            return False
    
    def process_faces(self, frame, rgb_frame):
        """
        Process faces in the frame using OpenCV's face recognition.
        
        Args:
            frame: Original BGR frame for drawing
            rgb_frame: RGB frame for processing
            
        Returns:
            frame: Frame with face boxes drawn
        """
        # Only process every other frame to save processing time
        if self.process_this_frame:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Reset current face
            self.current_face = None
            
            # Process each face found in the frame
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Resize to a standard size for the recognizer
                face_roi = cv2.resize(face_roi, (100, 100))
                
                # Save the current face if in naming mode
                if self.naming_mode and self.current_face is None:
                    self.current_face = face_roi
                
                # Try to recognize the face
                name = "Unknown"
                if len(self.known_faces) > 0:
                    try:
                        # Predict the face
                        label, confidence = self.face_recognizer.predict(face_roi)
                        
                        # If confidence is low enough, use the name (lower is better in LBPH)
                        if confidence < 70:  # Threshold can be adjusted
                            name = self.known_face_names[label]
                    except Exception as e:
                        print(f"Recognition error: {e}")
                
                # Draw a box around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Draw a label with a name below the face
                label = name
                if self.naming_mode and name == "Unknown":
                    label = "Press 'n' to name this face"
                
                cv2.rectangle(frame, (x, y+h), (x+w, y+h+35), (0, 0, 255), cv2.FILLED)
                cv2.putText(
                    frame,
                    label,
                    (x + 6, y+h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        self.process_this_frame = not self.process_this_frame
        return frame

    def run(self):
        """
        Run the vision recognition application.
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Print app status
        print("Vision Recognition App started. Press 'q' to quit.")
        print("Face recognition is ENABLED.")
        print("Press 'm' to toggle face naming mode.")
        print("Press 'n' to name a detected face when in naming mode.")
        
        print("\nGesture recognition is ENABLED. Supported gestures:")
        for gesture_name in self.gestures.values():
            print(f"- {gesture_name}")
        
        while cap.isOpened():
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Mirror the frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            frame = self.process_hands(frame, rgb_frame)
            
            # Process faces
            frame = self.process_faces(frame, rgb_frame)
            
            # Add app title
            cv2.putText(
                frame,
                "Vision Recognition App",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            
            # Add naming mode indicator
            if self.naming_mode:
                cv2.putText(
                    frame,
                    "Naming Mode: ON",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            
            # Display the frame
            cv2.imshow("Vision Recognition App", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on 'q' key press
            if key == ord('q'):
                break
            
            # Toggle naming mode on 'm' key press
            elif key == ord('m'):
                self.naming_mode = not self.naming_mode
                print(f"Naming mode: {'ON' if self.naming_mode else 'OFF'}")
            
            # Name the current face on 'n' key press when in naming mode
            elif key == ord('n') and self.naming_mode and self.current_face is not None:
                # Prompt for name
                cv2.putText(
                    frame,
                    "Enter name in terminal",
                    (frame.shape[1]//2 - 150, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Vision Recognition App", frame)
                
                # Get name from terminal
                print("\nEnter name for the detected face:")
                name = input("> ")
                
                if name.strip():
                    self.add_face(name.strip())
                    self.naming_mode = False  # Turn off naming mode after adding a face
                else:
                    print("Name cannot be empty. Face not added.")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create and run the Vision Recognition App
    app = VisionRecognitionApp()
    app.run()
