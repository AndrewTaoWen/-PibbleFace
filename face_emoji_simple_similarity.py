#!/usr/bin/env python3
"""
Simple Similarity Face Emoji Matcher - Just finds the most similar emoji without complex weighting.
"""

import cv2
import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

class SimpleSimilarityMatcher(FaceEmojiMatcherOpenCV):
    def __init__(self):
        """Initialize with simple similarity matching."""
        super().__init__()
        
        # Simplified emoji database with clear, distinct features
        self.emoji_database = {
            # Happy expressions
            '😊': {'smile_intensity': 0.7, 'eye_openness': 0.6, 'eyebrow_raise': 0.1, 'mouth_width': 0.5},
            '😄': {'smile_intensity': 0.9, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.7},
            '😃': {'smile_intensity': 0.8, 'eye_openness': 0.7, 'eyebrow_raise': 0.2, 'mouth_width': 0.6},
            '😁': {'smile_intensity': 1.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.8},
            '😆': {'smile_intensity': 0.9, 'eye_openness': 0.4, 'eyebrow_raise': 0.0, 'mouth_width': 0.7},
            '😅': {'smile_intensity': 0.6, 'eye_openness': 0.6, 'eyebrow_raise': 0.3, 'mouth_width': 0.4},
            '🤣': {'smile_intensity': 1.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.9},
            '😂': {'smile_intensity': 0.8, 'eye_openness': 0.1, 'eyebrow_raise': 0.0, 'mouth_width': 0.7},
            
            # Sad expressions
            '😭': {'smile_intensity': 0.0, 'eye_openness': 0.3, 'eyebrow_raise': 0.4, 'mouth_width': 0.3},
            '😢': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.3, 'mouth_width': 0.2},
            '😔': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.2, 'mouth_width': 0.1},
            '😕': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.1, 'mouth_width': 0.2},
            
            # Neutral expressions
            '😐': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😑': {'smile_intensity': 0.0, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '😒': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            '🙄': {'smile_intensity': 0.0, 'eye_openness': 0.2, 'eyebrow_raise': 0.0, 'mouth_width': 0.2},
            
            # Slight expressions
            '😏': {'smile_intensity': 0.3, 'eye_openness': 0.6, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '😎': {'smile_intensity': 0.4, 'eye_openness': 0.5, 'eyebrow_raise': 0.0, 'mouth_width': 0.3},
            '🤔': {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.3, 'mouth_width': 0.1},
            
            # Surprised expressions
            '😮': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.4, 'mouth_width': 0.5},
            '😯': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.2, 'mouth_width': 0.3},
            '😲': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.5, 'mouth_width': 0.4},
            '😳': {'smile_intensity': 0.2, 'eye_openness': 0.7, 'eyebrow_raise': 0.1, 'mouth_width': 0.2},
            
            # Fear expressions
            '😱': {'smile_intensity': 0.0, 'eye_openness': 0.9, 'eyebrow_raise': 0.7, 'mouth_width': 0.6},
            '😨': {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.5, 'mouth_width': 0.3},
            '😰': {'smile_intensity': 0.0, 'eye_openness': 0.7, 'eyebrow_raise': 0.3, 'mouth_width': 0.2},
            
            # Sleepy expressions
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
        """Calculate simple similarity - just find the closest match."""
        if not face_features:
            return 0.0
        
        # EQUAL WEIGHTS - no bias, just pure similarity
        weights = {
            'smile_intensity': 0.25,  # Equal weight
            'mouth_width': 0.25,      # Equal weight
            'eye_openness': 0.25,     # Equal weight
            'eyebrow_raise': 0.25     # Equal weight
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
    
    def find_best_emoji_match(self, face_features: dict) -> tuple:
        """Find the most similar emoji using simple similarity."""
        if not face_features:
            return "😐", 0.0
        
        best_emoji = "😐"
        best_similarity = 0.0
        
        # Check each emoji
        for emoji, emoji_features in self.emoji_database.items():
            similarity = self.calculate_similarity(face_features, emoji_features)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_emoji = emoji
        
        return best_emoji, best_similarity

def main():
    print("🎭 Face Emoji Matcher - Simple Similarity")
    print("=" * 50)
    print("✅ EQUAL WEIGHTS:")
    print("   - Smile: 25%")
    print("   - Mouth Width: 25%")
    print("   - Eye Openness: 25%")
    print("   - Eyebrow Raise: 25%")
    print("=" * 50)
    print("✅ Just finds the most similar emoji!")
    print("✅ No complex weighting or bias!")
    print("✅ Press 'q' to quit")
    print("=" * 50)
    
    matcher = SimpleSimilarityMatcher()
    
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
    
    print("✅ Starting simple similarity detection...")
    
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
        cv2.putText(display_frame, f"SIMILARITY: {similarity:.1%}", (50, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(display_frame, "SIMPLE MATCH", (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, "Press 'q' to quit", (50, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Update console
        if frame_count % 10 == 0:
            if emoji != last_emoji:
                print(f"🎭 New match: {emoji_name} ({emoji}) - similarity: {similarity:.3f}")
                last_emoji = emoji
        
        # Display
        cv2.imshow('Face Emoji Matcher - Simple Similarity', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Goodbye!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Simple similarity demo completed!")

if __name__ == "__main__":
    main()
