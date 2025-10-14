#!/usr/bin/env python3
"""
Simple text-based Face Emoji Matcher - Shows emoji names instead of symbols.
"""

import cv2
import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

def main():
    print("🎭 Face Emoji Matcher - Simple Text Version")
    print("=" * 50)
    print("This version shows emoji NAMES instead of symbols")
    print("to avoid the ??? display issue in OpenCV")
    print("=" * 50)
    
    # Initialize matcher
    matcher = FaceEmojiMatcherOpenCV()
    
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
    
    print("✅ Starting face detection...")
    print("Make different expressions to see emoji matches!")
    print("Press 'q' to quit, 's' to save current match")
    
    frame_count = 0
    last_emoji = "😐"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame, emoji, similarity = matcher.process_frame(frame)
        
        # Create a simple display
        display_frame = processed_frame.copy()
        
        # Add emoji name as text (this will work!)
        emoji_names = {
            '😊': 'HAPPY',
            '😄': 'VERY HAPPY', 
            '😃': 'EXCITED',
            '😁': 'GRINNING',
            '😆': 'LAUGHING',
            '😅': 'SWEAT SMILE',
            '🤣': 'ROFL',
            '😂': 'TEARS OF JOY',
            '😭': 'CRYING',
            '😢': 'SAD',
            '😔': 'PENSIVE',
            '😕': 'CONFUSED',
            '😐': 'NEUTRAL',
            '😑': 'EXPRESSIONLESS',
            '😒': 'UNAMUSED',
            '🙄': 'EYE ROLL',
            '😏': 'SMIRK',
            '😎': 'COOL',
            '🤔': 'THINKING',
            '😮': 'SURPRISED',
            '😯': 'HUSHED',
            '😲': 'ASTONISHED',
            '😳': 'FLUSHED',
            '😱': 'SCREAMING',
            '😨': 'FEARFUL',
            '😰': 'ANXIOUS',
            '😴': 'SLEEPING',
            '😪': 'SLEEPY',
            '🤤': 'DROOLING',
            '😋': 'YUMMY',
            '😍': 'HEART EYES',
            '🥰': 'SMILING HEARTS',
            '😘': 'KISS',
            '😗': 'KISSING',
            '😙': 'KISSING SMILE',
            '😚': 'KISSING CLOSED EYES',
            '🤗': 'HUGGING',
            '🤩': 'STAR EYES',
            '🤨': 'RAISED EYEBROW',
            '🧐': 'MONOCLE',
            '🤓': 'NERD',
            '😤': 'HUFFING',
            '😠': 'ANGRY',
            '😡': 'RAGING',
            '🤬': 'CURSING',
            '🤯': 'EXPLODING HEAD',
            '😵': 'DIZZY',
            '🥴': 'WOOZY',
            '😵‍💫': 'SPIRAL EYES',
            '🤢': 'NAUSEATED',
            '🤮': 'VOMITING',
            '🤧': 'SNEEZING',
            '🥵': 'HOT',
            '🥶': 'COLD',
            '😶': 'NO MOUTH',
            '😶‍🌫️': 'CLOUD',
            '😷': 'MASK',
            '🤒': 'SICK',
            '🤕': 'INJURED',
            '🤑': 'MONEY MOUTH',
            '🤠': 'COWBOY',
            '🥳': 'PARTY',
            '🥸': 'DISGUISE',
            '😈': 'SMILING DEVIL',
            '👿': 'ANGRY DEVIL',
            '👹': 'OGRE',
            '👺': 'GOBLIN',
            '💀': 'SKULL',
            '☠️': 'SKULL CROSSBONES',
            '💩': 'POOP',
            '🤡': 'CLOWN',
            '👻': 'GHOST',
            '👽': 'ALIEN',
            '👾': 'ALIEN MONSTER',
            '🤖': 'ROBOT',
            '😺': 'SMILING CAT',
            '😸': 'GRINNING CAT',
            '😹': 'JOY CAT',
            '😻': 'HEART EYES CAT',
            '😼': 'SMIRK CAT',
            '😽': 'KISSING CAT',
            '🙀': 'SCREAM CAT',
            '😿': 'CRYING CAT',
            '😾': 'POUTING CAT'
        }
        
        emoji_name = emoji_names.get(emoji, "UNKNOWN")
        
        # Draw emoji name (large and visible)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        color = (0, 255, 0)  # Green
        thickness = 3
        
        # Draw emoji name at the top
        cv2.putText(display_frame, emoji_name, (50, 80), font, font_scale, color, thickness)
        
        # Draw similarity score
        similarity_text = f"MATCH: {similarity:.1%}"
        cv2.putText(display_frame, similarity_text, (50, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Draw instructions
        cv2.putText(display_frame, "Press 'q' to quit, 's' to save", (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Update console output
        if frame_count % 10 == 0:
            if emoji != last_emoji:
                print(f"🎭 New match: {emoji_name} ({emoji}) - similarity: {similarity:.3f}")
                last_emoji = emoji
        
        # Display
        cv2.imshow('Face Emoji Matcher - Text Version', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Goodbye!")
            break
        elif key == ord('s'):
            print(f"💾 Saved: {emoji_name} ({emoji}) - similarity: {similarity:.3f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Demo completed!")

if __name__ == "__main__":
    main()
