import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple
import json
import os

class FaceEmojiMatcher:
    def __init__(self):
        """Initialize the face emoji matcher with MediaPipe face detection."""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define emoji database with facial characteristics
        self.emoji_database = {
            '😊': {
                'smile_intensity': 0.7,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.3,
                'mouth_width': 0.6
            },
            '😄': {
                'smile_intensity': 0.9,
                'eye_openness': 0.9,
                'eyebrow_raise': 0.2,
                'mouth_width': 0.8
            },
            '😃': {
                'smile_intensity': 0.8,
                'eye_openness': 1.0,
                'eyebrow_raise': 0.4,
                'mouth_width': 0.7
            },
            '😁': {
                'smile_intensity': 0.95,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.1,
                'mouth_width': 0.9
            },
            '😆': {
                'smile_intensity': 0.9,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.85
            },
            '😅': {
                'smile_intensity': 0.6,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.5,
                'mouth_width': 0.5
            },
            '🤣': {
                'smile_intensity': 1.0,
                'eye_openness': 0.3,
                'eyebrow_raise': 0.0,
                'mouth_width': 1.0
            },
            '😂': {
                'smile_intensity': 0.8,
                'eye_openness': 0.2,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.8
            },
            '😭': {
                'smile_intensity': 0.0,
                'eye_openness': 0.3,
                'eyebrow_raise': 0.8,
                'mouth_width': 0.4
            },
            '😢': {
                'smile_intensity': 0.0,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.6,
                'mouth_width': 0.3
            },
            '😔': {
                'smile_intensity': 0.0,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.4,
                'mouth_width': 0.2
            },
            '😕': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.3,
                'mouth_width': 0.3
            },
            '😐': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '😑': {
                'smile_intensity': 0.0,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😒': {
                'smile_intensity': 0.0,
                'eye_openness': 0.7,
                'eyebrow_raise': -0.2,
                'mouth_width': 0.3
            },
            '🙄': {
                'smile_intensity': 0.0,
                'eye_openness': 0.3,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😏': {
                'smile_intensity': 0.3,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '😎': {
                'smile_intensity': 0.4,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '🤔': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.6,
                'mouth_width': 0.2
            },
            '😮': {
                'smile_intensity': 0.0,
                'eye_openness': 1.0,
                'eyebrow_raise': 0.7,
                'mouth_width': 0.6
            },
            '😯': {
                'smile_intensity': 0.0,
                'eye_openness': 1.0,
                'eyebrow_raise': 0.5,
                'mouth_width': 0.4
            },
            '😲': {
                'smile_intensity': 0.0,
                'eye_openness': 1.0,
                'eyebrow_raise': 0.8,
                'mouth_width': 0.5
            },
            '😳': {
                'smile_intensity': 0.2,
                'eye_openness': 0.9,
                'eyebrow_raise': 0.3,
                'mouth_width': 0.3
            },
            '😱': {
                'smile_intensity': 0.0,
                'eye_openness': 1.0,
                'eyebrow_raise': 0.9,
                'mouth_width': 0.7
            },
            '😨': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': 0.7,
                'mouth_width': 0.4
            },
            '😰': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.6,
                'mouth_width': 0.3
            },
            '😴': {
                'smile_intensity': 0.0,
                'eye_openness': 0.0,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😪': {
                'smile_intensity': 0.0,
                'eye_openness': 0.2,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '🤤': {
                'smile_intensity': 0.0,
                'eye_openness': 0.3,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.6
            },
            '😋': {
                'smile_intensity': 0.6,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.5
            },
            '😍': {
                'smile_intensity': 0.7,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '🥰': {
                'smile_intensity': 0.8,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😘': {
                'smile_intensity': 0.5,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.2
            },
            '😗': {
                'smile_intensity': 0.4,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.1
            },
            '😙': {
                'smile_intensity': 0.6,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.2
            },
            '😚': {
                'smile_intensity': 0.5,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.1
            },
            '🤗': {
                'smile_intensity': 0.7,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '🤩': {
                'smile_intensity': 0.8,
                'eye_openness': 1.0,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.5
            },
            '🤨': {
                'smile_intensity': 0.0,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '🧐': {
                'smile_intensity': 0.0,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '🤓': {
                'smile_intensity': 0.3,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '😤': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': -0.3,
                'mouth_width': 0.3
            },
            '😠': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': -0.4,
                'mouth_width': 0.3
            },
            '😡': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': -0.5,
                'mouth_width': 0.4
            },
            '🤬': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': -0.6,
                'mouth_width': 0.5
            },
            '🤯': {
                'smile_intensity': 0.0,
                'eye_openness': 1.0,
                'eyebrow_raise': 0.8,
                'mouth_width': 0.6
            },
            '😵': {
                'smile_intensity': 0.0,
                'eye_openness': 0.3,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '🥴': {
                'smile_intensity': 0.3,
                'eye_openness': 0.4,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '😵‍💫': {
                'smile_intensity': 0.0,
                'eye_openness': 0.2,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '🤢': {
                'smile_intensity': 0.0,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '🤮': {
                'smile_intensity': 0.0,
                'eye_openness': 0.5,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '🤧': {
                'smile_intensity': 0.0,
                'eye_openness': 0.4,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '🥵': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '🥶': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😶': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.0
            },
            '😶‍🌫️': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.0
            },
            '😷': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.0
            },
            '🤒': {
                'smile_intensity': 0.0,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '🤕': {
                'smile_intensity': 0.0,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '🤑': {
                'smile_intensity': 0.6,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.5
            },
            '🤠': {
                'smile_intensity': 0.5,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '🥳': {
                'smile_intensity': 0.9,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.6
            },
            '🥸': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😈': {
                'smile_intensity': 0.7,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.5
            },
            '👿': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': -0.4,
                'mouth_width': 0.5
            },
            '👹': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': -0.5,
                'mouth_width': 0.6
            },
            '👺': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': -0.3,
                'mouth_width': 0.5
            },
            '💀': {
                'smile_intensity': 0.0,
                'eye_openness': 0.0,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '☠️': {
                'smile_intensity': 0.0,
                'eye_openness': 0.0,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '💩': {
                'smile_intensity': 0.5,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '🤡': {
                'smile_intensity': 0.8,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.6
            },
            '👻': {
                'smile_intensity': 0.0,
                'eye_openness': 0.0,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '👽': {
                'smile_intensity': 0.0,
                'eye_openness': 1.0,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.2
            },
            '👾': {
                'smile_intensity': 0.0,
                'eye_openness': 0.0,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '🤖': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😺': {
                'smile_intensity': 0.6,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '😸': {
                'smile_intensity': 0.8,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.5
            },
            '😹': {
                'smile_intensity': 0.9,
                'eye_openness': 0.3,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.6
            },
            '😻': {
                'smile_intensity': 0.7,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '😼': {
                'smile_intensity': 0.3,
                'eye_openness': 0.7,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😽': {
                'smile_intensity': 0.5,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.2
            },
            '🙀': {
                'smile_intensity': 0.0,
                'eye_openness': 1.0,
                'eyebrow_raise': 0.8,
                'mouth_width': 0.5
            },
            '😿': {
                'smile_intensity': 0.0,
                'eye_openness': 0.6,
                'eyebrow_raise': 0.6,
                'mouth_width': 0.3
            },
            '😾': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': -0.3,
                'mouth_width': 0.3
            }
        }

    def extract_facial_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract facial features from an image using MediaPipe."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get key facial points
        h, w, _ = image.shape
        
        # Eye landmarks
        left_eye_top = face_landmarks.landmark[159]
        left_eye_bottom = face_landmarks.landmark[145]
        right_eye_top = face_landmarks.landmark[386]
        right_eye_bottom = face_landmarks.landmark[374]
        
        # Eyebrow landmarks
        left_eyebrow = face_landmarks.landmark[70]
        right_eyebrow = face_landmarks.landmark[300]
        
        # Mouth landmarks
        mouth_left = face_landmarks.landmark[61]
        mouth_right = face_landmarks.landmark[291]
        mouth_top = face_landmarks.landmark[13]
        mouth_bottom = face_landmarks.landmark[14]
        
        # Calculate features
        # Eye openness (average of both eyes)
        left_eye_openness = abs(left_eye_top.y - left_eye_bottom.y) * h
        right_eye_openness = abs(right_eye_top.y - right_eye_bottom.y) * h
        eye_openness = (left_eye_openness + right_eye_openness) / 2
        eye_openness_normalized = min(eye_openness / 20, 1.0)  # Normalize to 0-1
        
        # Eyebrow raise (average height of both eyebrows)
        eyebrow_height = ((left_eyebrow.y + right_eyebrow.y) / 2) * h
        eyebrow_raise = max(0, (eyebrow_height - 100) / 50)  # Normalize
        eyebrow_raise = min(eyebrow_raise, 1.0)
        
        # Mouth width
        mouth_width = abs(mouth_right.x - mouth_left.x) * w
        mouth_width_normalized = min(mouth_width / 80, 1.0)  # Normalize
        
        # Smile intensity (mouth corner elevation)
        mouth_corners_y = (mouth_left.y + mouth_right.y) / 2
        mouth_center_y = (mouth_top.y + mouth_bottom.y) / 2
        smile_intensity = max(0, (mouth_center_y - mouth_corners_y) * h / 20)
        smile_intensity = min(smile_intensity, 1.0)
        
        return {
            'smile_intensity': smile_intensity,
            'eye_openness': eye_openness_normalized,
            'eyebrow_raise': eyebrow_raise,
            'mouth_width': mouth_width_normalized
        }

    def calculate_similarity(self, face_features: Dict[str, float], emoji_features: Dict[str, float]) -> float:
        """Calculate similarity between face features and emoji features."""
        if not face_features:
            return 0.0
        
        # Calculate weighted Euclidean distance
        weights = {
            'smile_intensity': 0.4,
            'eye_openness': 0.3,
            'eyebrow_raise': 0.2,
            'mouth_width': 0.1
        }
        
        total_distance = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in face_features and feature in emoji_features:
                diff = abs(face_features[feature] - emoji_features[feature])
                total_distance += weight * diff
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Convert distance to similarity (0-1, higher is more similar)
        similarity = 1.0 - (total_distance / total_weight)
        return max(0.0, similarity)

    def find_best_emoji_match(self, face_features: Dict[str, float]) -> Tuple[str, float]:
        """Find the best matching emoji for the given facial features."""
        if not face_features:
            return '😐', 0.0
        
        best_emoji = '😐'
        best_similarity = 0.0
        
        for emoji, emoji_features in self.emoji_database.items():
            similarity = self.calculate_similarity(face_features, emoji_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_emoji = emoji
        
        return best_emoji, best_similarity

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Process a single frame and return the frame with annotations and best emoji match."""
        # Extract facial features
        face_features = self.extract_facial_features(frame)
        
        # Find best emoji match
        best_emoji, similarity = self.find_best_emoji_match(face_features)
        
        # Draw face mesh on frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
        
        # Add emoji and similarity text to frame
        cv2.putText(frame, f"Best Match: {best_emoji}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Similarity: {similarity:.2f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, best_emoji, similarity

    def run_webcam_demo(self):
        """Run a real-time webcam demo of the face emoji matcher."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting face emoji matcher...")
        print("Press 'q' to quit, 's' to save current emoji")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, emoji, similarity = self.process_frame(frame)
            
            # Display frame
            cv2.imshow('Face Emoji Matcher', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"Saved: {emoji} (similarity: {similarity:.2f})")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    matcher = FaceEmojiMatcher()
    matcher.run_webcam_demo()
