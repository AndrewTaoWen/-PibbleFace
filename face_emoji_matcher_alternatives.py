import cv2
import numpy as np
from typing import Dict, List, Tuple
import json
import os
import platform

class FaceEmojiMatcherAlternatives:
    def __init__(self, detection_method="opencv"):
        """
        Initialize the face emoji matcher with different detection methods.
        
        Args:
            detection_method: "opencv", "dlib", or "mtcnn"
        """
        self.detection_method = detection_method
        
        if detection_method == "opencv":
            self._init_opencv()
        elif detection_method == "dlib":
            self._init_dlib()
        elif detection_method == "mtcnn":
            self._init_mtcnn()
        else:
            raise ValueError("detection_method must be 'opencv', 'dlib', or 'mtcnn'")
        
        # Define emoji database (same as before)
        self.emoji_database = {
            '😊': {'smile_intensity': 0.7, 'eye_openness': 0.8, 'eyebrow_raise': 0.3, 'mouth_width': 0.6},
            '😄': {'smile_intensity': 0.9, 'eye_openness': 0.9, 'eyebrow_raise': 0.2, 'mouth_width': 0.8},
            '😃': {'smile_intensity': 0.8, 'eye_openness': 1.0, 'eyebrow_raise': 0.4, 'mouth_width': 0.7},
            '😁': {'smile_intensity': 0.95, 'eye_openness': 0.7, 'eyebrow_raise': 0.1, 'mouth_width': 0.9},
            '😆': {'smile_intensity': 0.9, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.85},
            '😅': {'smile_intensity': 0.6, 'eye_openness': 0.8, 'eyebrow_raise': 0.5, 'mouth_width': 0.5},
            '🤣': {'smile_intensity': 1.0, 'eye_openness': 0.3, 'eyebrow_raise': 0.0, 'mouth_width': 1.0},
            '😂': {'smile_intensity': 0.8, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.8},
            '😭': {'smile_intensity': 0.0, 'eye_openness': 0.3, 'eyebrow_raise': 0.8, 'mouth_width': 0.4},
            '😢': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.6, 'mouth_width': 0.3},
            '😔': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.4, 'mouth_width': 0.2},
            '😕': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.3, 'mouth_width': 0.3},
            '😐': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😑': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😒': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🙄': {'smile_intensity': 0.0, 'eye_openness': 0.3, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😏': {'smile_intensity': 0.3, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😎': {'smile_intensity': 0.4, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤔': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.6, 'mouth_width': 0.2},
            '😮': {'smile_intensity': 0.0, 'eye_openness': 1.0, 'eyebrow_raise': 0.7, 'mouth_width': 0.6},
            '😯': {'smile_intensity': 0.0, 'eye_openness': 1.0, 'eyebrow_raise': 0.5, 'mouth_width': 0.4},
            '😲': {'smile_intensity': 0.0, 'eye_openness': 1.0, 'eyebrow_raise': 0.8, 'mouth_width': 0.5},
            '😳': {'smile_intensity': 0.2, 'eye_openness': 0.9, 'eyebrow_raise': 0.3, 'mouth_width': 0.3},
            '😱': {'smile_intensity': 0.0, 'eye_openness': 1.0, 'eyebrow_raise': 0.9, 'mouth_width': 0.7},
            '😨': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.7, 'mouth_width': 0.4},
            '😰': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.6, 'mouth_width': 0.3},
            '😴': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😪': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤤': {'smile_intensity': 0.0, 'eye_openness': 0.3, 'eyebrow_raise': 0.0, 'mouth_width': 0.6},
            '😋': {'smile_intensity': 0.6, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '😍': {'smile_intensity': 0.7, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🥰': {'smile_intensity': 0.8, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😘': {'smile_intensity': 0.5, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😗': {'smile_intensity': 0.4, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '😙': {'smile_intensity': 0.6, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😚': {'smile_intensity': 0.5, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '🤗': {'smile_intensity': 0.7, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤩': {'smile_intensity': 0.8, 'eye_openness': 1.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '🤨': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🧐': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤓': {'smile_intensity': 0.3, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😤': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😠': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😡': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤬': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '🤯': {'smile_intensity': 0.0, 'eye_openness': 1.0, 'eyebrow_raise': 0.8, 'mouth_width': 0.6},
            '😵': {'smile_intensity': 0.0, 'eye_openness': 0.3, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🥴': {'smile_intensity': 0.3, 'eye_openness': 0.4, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😵‍💫': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤢': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤮': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤧': {'smile_intensity': 0.0, 'eye_openness': 0.4, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🥵': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🥶': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😶': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.0},
            '😶‍🌫️': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.0},
            '😷': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.0},
            '🤒': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤕': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤑': {'smile_intensity': 0.6, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '🤠': {'smile_intensity': 0.5, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🥳': {'smile_intensity': 0.9, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.6},
            '🥸': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😈': {'smile_intensity': 0.7, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '👿': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '👹': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.0, 'mouth_width': 0.6},
            '👺': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '💀': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '☠️': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '💩': {'smile_intensity': 0.5, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤡': {'smile_intensity': 0.8, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.6},
            '👻': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '👽': {'smile_intensity': 0.0, 'eye_openness': 1.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '👾': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤖': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😺': {'smile_intensity': 0.6, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😸': {'smile_intensity': 0.8, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '😹': {'smile_intensity': 0.9, 'eye_openness': 0.3, 'eyebrow_raise': 0.0, 'mouth_width': 0.6},
            '😻': {'smile_intensity': 0.7, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😼': {'smile_intensity': 0.3, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😽': {'smile_intensity': 0.5, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🙀': {'smile_intensity': 0.0, 'eye_openness': 1.0, 'eyebrow_raise': 0.8, 'mouth_width': 0.5},
            '😿': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.6, 'mouth_width': 0.3},
            '😾': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.3}
        }

    def _init_opencv(self):
        """Initialize OpenCV-based face detection."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        print("✅ OpenCV face detection initialized")

    def _init_dlib(self):
        """Initialize dlib-based face detection."""
        try:
            import dlib
            # You would need to download the shape predictor file
            # self.detector = dlib.get_frontal_face_detector()
            # self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            print("⚠️  dlib not available - falling back to OpenCV")
            self._init_opencv()
            self.detection_method = "opencv"
        except ImportError:
            print("⚠️  dlib not installed - falling back to OpenCV")
            self._init_opencv()
            self.detection_method = "opencv"

    def _init_mtcnn(self):
        """Initialize MTCNN-based face detection."""
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            print("✅ MTCNN face detection initialized")
        except ImportError:
            print("⚠️  MTCNN not installed - falling back to OpenCV")
            self._init_opencv()
            self.detection_method = "opencv"

    def extract_facial_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract facial features using the selected detection method."""
        if self.detection_method == "opencv":
            return self._extract_features_opencv(image)
        elif self.detection_method == "dlib":
            return self._extract_features_dlib(image)
        elif self.detection_method == "mtcnn":
            return self._extract_features_mtcnn(image)
        else:
            return None

    def _extract_features_opencv(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features using OpenCV."""
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
        eye_openness = min(len(eyes) / 2.0, 1.0)
        smile_intensity = min(len(smiles) / 1.0, 1.0)
        eyebrow_raise = max(0, (y - 100) / 100)
        eyebrow_raise = min(eyebrow_raise, 1.0)
        mouth_width = min(w / 100, 1.0)
        
        return {
            'smile_intensity': smile_intensity,
            'eye_openness': eye_openness,
            'eyebrow_raise': eyebrow_raise,
            'mouth_width': mouth_width
        }

    def _extract_features_dlib(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features using dlib (placeholder)."""
        # This would require dlib and shape predictor
        return None

    def _extract_features_mtcnn(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features using MTCNN."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.detector.detect_faces(rgb_image)
            
            if len(faces) == 0:
                return None
            
            # Use the face with highest confidence
            face = max(faces, key=lambda x: x['confidence'])
            
            # Extract key points
            keypoints = face['keypoints']
            
            # Calculate features based on keypoints
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            nose = keypoints['nose']
            mouth_left = keypoints['mouth_left']
            mouth_right = keypoints['mouth_right']
            
            # Calculate eye openness (distance between eyes)
            eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
            eye_openness = min(eye_distance / 50, 1.0)
            
            # Calculate smile intensity (mouth width)
            mouth_width = np.sqrt((mouth_right[0] - mouth_left[0])**2 + (mouth_right[1] - mouth_left[1])**2)
            smile_intensity = min(mouth_width / 30, 1.0)
            
            # Calculate eyebrow raise (eye to nose distance)
            eyebrow_raise = max(0, (nose[1] - (left_eye[1] + right_eye[1])/2) / 20)
            eyebrow_raise = min(eyebrow_raise, 1.0)
            
            # Mouth width (normalized)
            mouth_width_norm = min(mouth_width / 40, 1.0)
            
            return {
                'smile_intensity': smile_intensity,
                'eye_openness': eye_openness,
                'eyebrow_raise': eyebrow_raise,
                'mouth_width': mouth_width_norm
            }
        except Exception as e:
            print(f"Error in MTCNN detection: {e}")
            return None

    def calculate_similarity(self, face_features: Dict[str, float], emoji_features: Dict[str, float]) -> float:
        """Calculate similarity between face features and emoji features."""
        if not face_features:
            return 0.0
        
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
        
        # Draw detection results
        if self.detection_method == "opencv":
            self._draw_opencv_detections(frame)
        elif self.detection_method == "mtcnn":
            self._draw_mtcnn_detections(frame)
        
        # Add emoji and similarity text to frame
        cv2.putText(frame, f"Best Match: {best_emoji}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Similarity: {similarity:.2f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Method: {self.detection_method.upper()}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, best_emoji, similarity

    def _draw_opencv_detections(self, frame: np.ndarray):
        """Draw OpenCV detection results."""
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

    def _draw_mtcnn_detections(self, frame: np.ndarray):
        """Draw MTCNN detection results."""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb_image)
            
            for face in faces:
                # Draw bounding box
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw keypoints
                keypoints = face['keypoints']
                for point in keypoints.values():
                    cv2.circle(frame, tuple(point), 3, (255, 0, 0), -1)
        except Exception as e:
            print(f"Error drawing MTCNN detections: {e}")

    def run_webcam_demo(self):
        """Run a real-time webcam demo."""
        print(f"🎭 Face Emoji Matcher - {self.detection_method.upper()} Demo")
        print("=" * 60)
        print("Instructions:")
        print("- Position your face in front of the camera")
        print("- Make different facial expressions")
        print("- Press 'q' to quit, 's' to save current emoji")
        print("- Press 'm' to switch detection method")
        print("=" * 60)
        
        # Try different camera indices for macOS
        camera_indices = [0, 1, 2, -1]  # -1 for default camera
        
        cap = None
        for camera_idx in camera_indices:
            print(f"🔄 Trying camera index {camera_idx}...")
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                print(f"✅ Camera {camera_idx} opened successfully!")
                break
            else:
                cap.release()
        
        if not cap or not cap.isOpened():
            print("❌ Error: Could not open any camera")
            print("Make sure your camera is connected and not being used by another application.")
            print("On macOS, you may need to grant camera permissions in System Preferences.")
            return
        
        frame_count = 0
        last_emoji = "😐"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
            
            # Process the frame
            processed_frame, emoji, similarity = self.process_frame(frame)
            
            # Update display every 10 frames to reduce flickering
            if frame_count % 10 == 0:
                if emoji != last_emoji:
                    print(f"🎭 New match: {emoji} (similarity: {similarity:.2f})")
                    last_emoji = emoji
            
            # Display the frame
            cv2.imshow(f'Face Emoji Matcher - {self.detection_method.upper()}', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Goodbye!")
                break
            elif key == ord('s'):
                print(f"💾 Saved: {emoji} (similarity: {similarity:.2f})")
            elif key == ord('m'):
                print("🔄 Switching detection method...")
                if self.detection_method == "opencv":
                    self.detection_method = "mtcnn"
                    self._init_mtcnn()
                elif self.detection_method == "mtcnn":
                    self.detection_method = "opencv"
                    self._init_opencv()
                print(f"✅ Switched to {self.detection_method.upper()}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("🎉 Demo completed!")

if __name__ == "__main__":
    # Try different detection methods
    print("Available detection methods:")
    print("1. OpenCV (default, works everywhere)")
    print("2. MTCNN (more accurate, requires installation)")
    
    choice = input("Choose detection method (1 or 2, default 1): ").strip()
    
    if choice == "2":
        try:
            matcher = FaceEmojiMatcherAlternatives("mtcnn")
        except:
            print("⚠️  MTCNN not available, using OpenCV")
            matcher = FaceEmojiMatcherAlternatives("opencv")
    else:
        matcher = FaceEmojiMatcherAlternatives("opencv")
    
    matcher.run_webcam_demo()
