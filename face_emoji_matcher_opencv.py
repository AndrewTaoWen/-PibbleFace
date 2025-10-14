import cv2
import numpy as np
from typing import Dict, List, Tuple
import json
import os

class FaceEmojiMatcherOpenCV:
    def __init__(self):
        """Initialize the face emoji matcher with OpenCV face detection."""
        # Load OpenCV's Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
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
                'eyebrow_raise': 0.0,
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
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😠': {
                'smile_intensity': 0.0,
                'eye_openness': 0.8,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            },
            '😡': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.4
            },
            '🤬': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': 0.0,
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
                'eyebrow_raise': 0.0,
                'mouth_width': 0.5
            },
            '👹': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': 0.0,
                'mouth_width': 0.6
            },
            '👺': {
                'smile_intensity': 0.0,
                'eye_openness': 0.9,
                'eyebrow_raise': 0.0,
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
                'eyebrow_raise': 0.0,
                'mouth_width': 0.3
            }
        }

    def extract_facial_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract facial features from an image using OpenCV."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Use the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 4)
        
        # Detect smiles
        smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
        
        # Calculate features
        # Eye openness (based on number of detected eyes)
        eye_openness = min(len(eyes) / 2.0, 1.0)  # Normalize to 0-1
        
        # Smile intensity (based on number of detected smiles)
        smile_intensity = min(len(smiles) / 1.0, 1.0)  # Normalize to 0-1
        
        # Eyebrow raise (simplified - based on face position)
        eyebrow_raise = max(0, (y - 100) / 100)  # Simple approximation
        eyebrow_raise = min(eyebrow_raise, 1.0)
        
        # Mouth width (based on face width)
        mouth_width = min(w / 100, 1.0)  # Normalize to 0-1
        
        return {
            'smile_intensity': smile_intensity,
            'eye_openness': eye_openness,
            'eyebrow_raise': eyebrow_raise,
            'mouth_width': mouth_width
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
        
        # Draw face detection rectangles
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw eyes
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 4)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
            
            # Draw smiles
            smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 0, 255), 2)
        
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
            cv2.imshow('Face Emoji Matcher (OpenCV)', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"Saved: {emoji} (similarity: {similarity:.2f})")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    matcher = FaceEmojiMatcherOpenCV()
    matcher.run_webcam_demo()
