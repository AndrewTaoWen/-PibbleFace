#!/usr/bin/env python3
"""
Adjusted Face Emoji Matcher - More reasonable thresholds to avoid defaulting to screaming.
"""

import cv2
import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

class AdjustedFaceEmojiMatcher(FaceEmojiMatcherOpenCV):
    def __init__(self):
        """Initialize with adjusted thresholds."""
        super().__init__()
        
        # Adjust the emoji database to be more reasonable
        self.emoji_database = {
            '😊': {'smile_intensity': 0.6, 'eye_openness': 0.7, 'eyebrow_raise': 0.2, 'mouth_width': 0.5},
            '😄': {'smile_intensity': 0.8, 'eye_openness': 0.8, 'eyebrow_raise': 0.1, 'mouth_width': 0.7},
            '😃': {'smile_intensity': 0.7, 'eye_openness': 0.9, 'eyebrow_raise': 0.3, 'mouth_width': 0.6},
            '😁': {'smile_intensity': 0.9, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.8},
            '😆': {'smile_intensity': 0.8, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.7},
            '😅': {'smile_intensity': 0.5, 'eye_openness': 0.7, 'eyebrow_raise': 0.4, 'mouth_width': 0.4},
            '🤣': {'smile_intensity': 1.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.9},
            '😂': {'smile_intensity': 0.7, 'eye_openness': 0.1, 'eyebrow_raise': 0.0, 'mouth_width': 0.7},
            '😭': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.7, 'mouth_width': 0.3},
            '😢': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.5, 'mouth_width': 0.2},
            '😔': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.3, 'mouth_width': 0.1},
            '😕': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.2, 'mouth_width': 0.2},
            '😐': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😑': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😒': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🙄': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😏': {'smile_intensity': 0.2, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😎': {'smile_intensity': 0.3, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤔': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.5, 'mouth_width': 0.1},
            '😮': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.6, 'mouth_width': 0.5},
            '😯': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.4, 'mouth_width': 0.3},
            '😲': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.7, 'mouth_width': 0.4},
            '😳': {'smile_intensity': 0.1, 'eye_openness': 0.8, 'eyebrow_raise': 0.2, 'mouth_width': 0.2},
            '😱': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.8, 'mouth_width': 0.6},
            '😨': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.6, 'mouth_width': 0.3},
            '😰': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.5, 'mouth_width': 0.2},
            '😴': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😪': {'smile_intensity': 0.0, 'eye_openness': 0.1, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤤': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '😋': {'smile_intensity': 0.5, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😍': {'smile_intensity': 0.6, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🥰': {'smile_intensity': 0.7, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😘': {'smile_intensity': 0.4, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '😗': {'smile_intensity': 0.3, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '😙': {'smile_intensity': 0.5, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '😚': {'smile_intensity': 0.4, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '🤗': {'smile_intensity': 0.6, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤩': {'smile_intensity': 0.7, 'eye_openness': 0.9, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤨': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🧐': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤓': {'smile_intensity': 0.2, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😤': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😠': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😡': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤬': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤯': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.7, 'mouth_width': 0.5},
            '😵': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🥴': {'smile_intensity': 0.2, 'eye_openness': 0.3, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😵‍💫': {'smile_intensity': 0.0, 'eye_openness': 0.1, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤢': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤮': {'smile_intensity': 0.0, 'eye_openness': 0.4, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤧': {'smile_intensity': 0.0, 'eye_openness': 0.3, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🥵': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🥶': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😶': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.0},
            '😶‍🌫️': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.0},
            '😷': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.0},
            '🤒': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤕': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤑': {'smile_intensity': 0.5, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '🤠': {'smile_intensity': 0.4, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🥳': {'smile_intensity': 0.8, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '🥸': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😈': {'smile_intensity': 0.6, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '👿': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '👹': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '👺': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '💀': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '☠️': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '💩': {'smile_intensity': 0.4, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤡': {'smile_intensity': 0.7, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '👻': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '👽': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '👾': {'smile_intensity': 0.0, 'eye_openness': 0.0, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🤖': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😺': {'smile_intensity': 0.5, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😸': {'smile_intensity': 0.7, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.4},
            '😹': {'smile_intensity': 0.8, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.5},
            '😻': {'smile_intensity': 0.6, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😼': {'smile_intensity': 0.2, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😽': {'smile_intensity': 0.4, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.1},
            '🙀': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.7, 'mouth_width': 0.4},
            '😿': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.5, 'mouth_width': 0.2},
            '😾': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.0, 'mouth_width': 0.2}
        }
    
    def extract_facial_features(self, image: np.ndarray):
        """Extract features with more conservative thresholds."""
        features = super().extract_facial_features(image)
        
        if features is None:
            return None
        
        # Apply more conservative thresholds
        # Cap extreme values that might cause false matches
        features['eye_openness'] = min(features['eye_openness'], 0.8)  # Cap at 0.8 instead of 1.0
        features['eyebrow_raise'] = min(features['eyebrow_raise'], 0.6)  # Cap at 0.6 instead of 1.0
        features['smile_intensity'] = min(features['smile_intensity'], 0.9)  # Cap at 0.9
        features['mouth_width'] = min(features['mouth_width'], 0.8)  # Cap at 0.8
        
        return features

def main():
    print("🎭 Face Emoji Matcher - Adjusted Thresholds")
    print("=" * 50)
    print("This version has more reasonable thresholds")
    print("to avoid defaulting to extreme expressions")
    print("=" * 50)
    
    matcher = AdjustedFaceEmojiMatcher()
    
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
    
    print("✅ Starting adjusted face detection...")
    
    frame_count = 0
    last_emoji = "😐"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame, emoji, similarity = matcher.process_frame(frame)
        
        # Create simple display
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
        cv2.putText(display_frame, "Press 'q' to quit", (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Update console
        if frame_count % 10 == 0:
            if emoji != last_emoji:
                print(f"🎭 New match: {emoji_name} ({emoji}) - similarity: {similarity:.3f}")
                last_emoji = emoji
        
        # Display
        cv2.imshow('Face Emoji Matcher - Adjusted', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Goodbye!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Adjusted demo completed!")

if __name__ == "__main__":
    main()
