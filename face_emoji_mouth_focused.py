#!/usr/bin/env python3
"""
Mouth-focused Face Emoji Matcher - Gives more weight to mouth expressions and reduces eye/eyebrow weight.
"""

import cv2
import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

class MouthFocusedFaceEmojiMatcher(FaceEmojiMatcherOpenCV):
    def __init__(self):
        """Initialize with mouth-focused weights."""
        super().__init__()
        
        # Adjust emoji database to be more mouth-focused
        self.emoji_database = {
            # Happy expressions (high smile, normal eyes)
            '😊': {'smile_intensity': 0.7, 'eye_openness': 0.6, 'eyebrow_raise': 0.1, 'mouth_width': 0.5},
            '😄': {'smile_intensity': 0.9, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.7},
            '😃': {'smile_intensity': 0.8, 'eye_openness': 0.7, 'eyebrow_raise': 0.2, 'mouth_width': 0.6},
            '😁': {'smile_intensity': 1.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.8},
            '😆': {'smile_intensity': 0.9, 'eye_openness': 0.4, 'eyebrow_raise': 0.0, 'mouth_width': 0.7},
            '😅': {'smile_intensity': 0.6, 'eye_openness': 0.6, 'eyebrow_raise': 0.3, 'mouth_width': 0.4},
            '🤣': {'smile_intensity': 1.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.9},
            '😂': {'smile_intensity': 0.8, 'eye_openness': 0.1, 'eyebrow_raise': 0.0, 'mouth_width': 0.7},
            
            # Sad expressions (no smile, normal eyes)
            '😭': {'smile_intensity': 0.0, 'eye_openness': 0.3, 'eyebrow_raise': 0.4, 'mouth_width': 0.3},
            '😢': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.3, 'mouth_width': 0.2},
            '😔': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.2, 'mouth_width': 0.1},
            '😕': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.1, 'mouth_width': 0.2},
            
            # Neutral expressions (no smile, normal eyes)
            '😐': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😑': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😒': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🙄': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            
            # Slight expressions (low smile, normal eyes)
            '😏': {'smile_intensity': 0.3, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😎': {'smile_intensity': 0.4, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤔': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.3, 'mouth_width': 0.1},
            
            # Surprised expressions (no smile, wide eyes)
            '😮': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.4, 'mouth_width': 0.5},
            '😯': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.2, 'mouth_width': 0.3},
            '😲': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.5, 'mouth_width': 0.4},
            '😳': {'smile_intensity': 0.2, 'eye_openness': 0.7, 'eyebrow_raise': 0.1, 'mouth_width': 0.2},
            
            # Fear expressions (no smile, wide eyes, high eyebrows)
            '😱': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.7, 'mouth_width': 0.6},
            '😨': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.5, 'mouth_width': 0.3},
            '😰': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.3, 'mouth_width': 0.2},
            
            # Sleepy expressions (no smile, closed eyes)
            '😴': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😪': {'smile_intensity': 0.0, 'eye_openness': 0.1, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            
            # Mouth-focused expressions
            '🤤': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '😋': {'smile_intensity': 0.6, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😍': {'smile_intensity': 0.7, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🥰': {'smile_intensity': 0.8, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😘': {'smile_intensity': 0.5, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '😗': {'smile_intensity': 0.4, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '😙': {'smile_intensity': 0.6, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '😚': {'smile_intensity': 0.5, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '🤗': {'smile_intensity': 0.7, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤩': {'smile_intensity': 0.8, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            
            # Other expressions
            '🤨': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🧐': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤓': {'smile_intensity': 0.3, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😤': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😠': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😡': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤬': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤯': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.5, 'mouth_width': 0.5},
            '😵': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🥴': {'smile_intensity': 0.3, 'eye_openness': 0.3, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😵‍💫': {'smile_intensity': 0.0, 'eye_openness': 0.1, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤢': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤮': {'smile_intensity': 0.0, 'eye_openness': 0.4, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤧': {'smile_intensity': 0.0, 'eye_openness': 0.3, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🥵': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🥶': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😶': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.0},
            '😶‍🌫️': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.0},
            '😷': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.0},
            '🤒': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤕': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤑': {'smile_intensity': 0.6, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤠': {'smile_intensity': 0.5, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🥳': {'smile_intensity': 0.9, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '🥸': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😈': {'smile_intensity': 0.7, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '👿': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '👹': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '👺': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '💀': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '☠️': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '💩': {'smile_intensity': 0.5, 'eye_openness': 0.4, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤡': {'smile_intensity': 0.8, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '👻': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '👽': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '👾': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤖': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😺': {'smile_intensity': 0.6, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😸': {'smile_intensity': 0.8, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😹': {'smile_intensity': 0.9, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '😻': {'smile_intensity': 0.7, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😼': {'smile_intensity': 0.3, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😽': {'smile_intensity': 0.5, 'eye_openness': 0.4, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '🙀': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.5, 'mouth_width': 0.4},
            '😿': {'smile_intensity': 0.0, 'eye_openness': 0.4, 'eyebrow_raise': 0.3, 'mouth_width': 0.2},
            '😾': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2}
        }
    
    def calculate_similarity(self, face_features: dict, emoji_features: dict) -> float:
        """Calculate similarity with mouth-focused weights."""
        if not face_features:
            return 0.0
        
        # NEW WEIGHTS: Much more focus on mouth, less on eyes/eyebrows
        weights = {
            'smile_intensity': 0.6,  # Increased from 0.4 to 0.6
            'mouth_width': 0.3,      # Increased from 0.1 to 0.3
            'eye_openness': 0.05,    # Decreased from 0.3 to 0.05
            'eyebrow_raise': 0.05    # Decreased from 0.2 to 0.05
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
    
    def extract_facial_features(self, image: np.ndarray):
        """Extract features with mouth-focused adjustments."""
        features = super().extract_facial_features(image)
        
        if features is None:
            return None
        
        # Apply mouth-focused adjustments
        # Boost mouth-related features, cap eye-related features
        features['smile_intensity'] = min(features['smile_intensity'] * 1.2, 1.0)  # Boost smile detection
        features['mouth_width'] = min(features['mouth_width'] * 1.1, 1.0)  # Boost mouth width
        features['eye_openness'] = min(features['eye_openness'], 0.7)  # Cap eye openness
        features['eyebrow_raise'] = min(features['eyebrow_raise'], 0.5)  # Cap eyebrow raise
        
        return features

def main():
    print("🎭 Face Emoji Matcher - Mouth Focused")
    print("=" * 50)
    print("✅ NEW WEIGHTS:")
    print("   - Smile: 60% (was 40%)")
    print("   - Mouth Width: 30% (was 10%)")
    print("   - Eye Openness: 5% (was 30%)")
    print("   - Eyebrow Raise: 5% (was 20%)")
    print("=" * 50)
    print("This should give much more reasonable results!")
    print("Press 'q' to quit")
    print("=" * 50)
    
    matcher = MouthFocusedFaceEmojiMatcher()
    
    # Find working camera
    camera_indices = [0, 1, 2, -1]
    cap = None
    
    for camera_idx in camera_indices:
        print(f"🔄 Trying camera index {camera_idx}...")
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {camera_idx} is working!")
                break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("❌ Error: Could not open any camera")
        return
    
    print("✅ Starting mouth-focused face detection...")
    
    frame_count = 0
    last_emoji = "😐"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame, emoji, similarity = matcher.process_frame(frame)
        
        # Create display
        display_frame = processed_frame.copy()
        
        # Add emoji name
        emoji_names = {
            '😊': 'HAPPY', '😄': 'VERY HAPPY', '😃': 'EXCITED', '😁': 'GRINNING',
            '😆': 'LAUGHING', '😅': 'SWEAT SMILE', '🤣': 'ROFL', '😂': 'TEARS OF JOY',
            '😭': 'CRYING', '😢': 'SAD', '😔': 'PENSIVE', '😕': 'CONFUSED',
            '😐': 'NEUTRAL', '😑': 'EXPRESSIONLESS', '😒': 'UNAMUSED', '🙄': 'EYE ROLL',
            '😏': 'SMIRK', '😎': 'COOL', '🤔': 'THINKING', '😮': 'SURPRISED',
            '😯': 'HUSHED', '😲': 'ASTONISHED', '😳': 'FLUSHED', '😱': 'SCREAMING',
            '😨': 'FEARFUL', '😰': 'ANXIOUS', '😴': 'SLEEPING', '😪': 'SLEEPY',
            '🤤': 'DROOLING', '😋': 'YUMMY', '😍': 'HEART EYES', '🥰': 'SMILING HEARTS',
            '😘': 'KISS', '😗': 'KISSING', '😙': 'KISSING SMILE', '😚': 'KISSING CLOSED EYES',
            '🤗': 'HUGGING', '🤩': 'STAR EYES', '🤨': 'RAISED EYEBROW', '🧐': 'MONOCLE',
            '🤓': 'NERD', '😤': 'HUFFING', '😠': 'ANGRY', '😡': 'RAGING',
            '🤬': 'CURSING', '🤯': 'EXPLODING HEAD', '😵': 'DIZZY', '🥴': 'WOOZY',
            '😵‍💫': 'SPIRAL EYES', '🤢': 'NAUSEATED', '🤮': 'VOMITING', '🤧': 'SNEEZING',
            '🥵': 'HOT', '🥶': 'COLD', '😶': 'NO MOUTH', '😶‍🌫️': 'CLOUD',
            '😷': 'MASK', '🤒': 'SICK', '🤕': 'INJURED', '🤑': 'MONEY MOUTH',
            '🤠': 'COWBOY', '🥳': 'PARTY', '🥸': 'DISGUISE', '😈': 'SMILING DEVIL',
            '👿': 'ANGRY DEVIL', '👹': 'OGRE', '👺': 'GOBLIN', '💀': 'SKULL',
            '☠️': 'SKULL CROSSBONES', '💩': 'POOP', '🤡': 'CLOWN', '👻': 'GHOST',
            '👽': 'ALIEN', '👾': 'ALIEN MONSTER', '🤖': 'ROBOT', '😺': 'SMILING CAT',
            '😸': 'GRINNING CAT', '😹': 'JOY CAT', '😻': 'HEART EYES CAT', '😼': 'SMIRK CAT',
            '😽': 'KISSING CAT', '🙀': 'SCREAM CAT', '😿': 'CRYING CAT', '😾': 'POUTING CAT'
        }
        
        emoji_name = emoji_names.get(emoji, "UNKNOWN")
        
        # Draw emoji name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        color = (0, 255, 0)
        thickness = 3
        
        cv2.putText(display_frame, emoji_name, (50, 80), font, font_scale, color, thickness)
        cv2.putText(display_frame, f"MATCH: {similarity:.1%}", (50, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(display_frame, "MOUTH FOCUSED", (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, "Press 'q' to quit", (50, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Update console
        if frame_count % 10 == 0:
            if emoji != last_emoji:
                print(f"🎭 New match: {emoji_name} ({emoji}) - similarity: {similarity:.3f}")
                last_emoji = emoji
        
        # Display
        cv2.imshow('Face Emoji Matcher - Mouth Focused', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Goodbye!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Mouth-focused demo completed!")

if __name__ == "__main__":
    main()
