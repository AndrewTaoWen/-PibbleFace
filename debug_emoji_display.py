#!/usr/bin/env python3
"""
Debug version to test emoji display and similarity thresholds.
"""

import cv2
import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

def test_emoji_display():
    """Test if emojis can be displayed in OpenCV."""
    print("🧪 Testing emoji display in OpenCV...")
    
    # Create a test image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Test different emojis
    emojis = ['😊', '😄', '😃', '😁', '😆', '😅', '🤣', '😂', '😭', '😢', '😔', '😕', '😐']
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    color = (0, 0, 0)
    thickness = 3
    
    y_pos = 50
    for i, emoji in enumerate(emojis):
        x_pos = 50 + (i % 3) * 150
        if i % 3 == 0 and i > 0:
            y_pos += 80
        
        # Try to draw the emoji
        try:
            cv2.putText(img, emoji, (x_pos, y_pos), font, font_scale, color, thickness)
            print(f"✅ Successfully drew: {emoji}")
        except Exception as e:
            print(f"❌ Failed to draw {emoji}: {e}")
    
    # Show the image
    cv2.imshow('Emoji Display Test', img)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_similarity_thresholds():
    """Test different similarity thresholds."""
    print("\n🧪 Testing similarity thresholds...")
    
    matcher = FaceEmojiMatcherOpenCV()
    
    # Test with various face features
    test_cases = [
        ("Happy", {'smile_intensity': 0.8, 'eye_openness': 0.9, 'eyebrow_raise': 0.2, 'mouth_width': 0.7}),
        ("Sad", {'smile_intensity': 0.0, 'eye_openness': 0.6, 'eyebrow_raise': 0.6, 'mouth_width': 0.3}),
        ("Surprised", {'smile_intensity': 0.0, 'eye_openness': 1.0, 'eyebrow_raise': 0.8, 'mouth_width': 0.6}),
        ("Neutral", {'smile_intensity': 0.0, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.4}),
        ("Slight Smile", {'smile_intensity': 0.3, 'eye_openness': 0.8, 'eyebrow_raise': 0.0, 'mouth_width': 0.5}),
    ]
    
    for name, features in test_cases:
        best_emoji, similarity = matcher.find_best_emoji_match(features)
        print(f"{name:12}: {best_emoji} (similarity: {similarity:.3f})")

def run_debug_demo():
    """Run a debug demo with lower similarity threshold."""
    print("\n🎭 Running debug demo with relaxed thresholds...")
    
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
    
    print("✅ Starting debug demo...")
    print("This version will show emojis even with low similarity scores")
    
    frame_count = 0
    last_emoji = "😐"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame, emoji, similarity = matcher.process_frame(frame)
        
        # Create a simple display with emoji
        display_frame = processed_frame.copy()
        
        # Add emoji text (larger and more visible)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3.0
        color = (0, 255, 0)
        thickness = 4
        
        # Draw emoji at the top
        cv2.putText(display_frame, emoji, (50, 100), font, font_scale, color, thickness)
        
        # Draw similarity score
        similarity_text = f"Match: {similarity:.2f}"
        cv2.putText(display_frame, similarity_text, (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Draw instructions
        cv2.putText(display_frame, "Press 'q' to quit", (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Update display
        if frame_count % 10 == 0:
            if emoji != last_emoji:
                print(f"🎭 New match: {emoji} (similarity: {similarity:.3f})")
                last_emoji = emoji
        
        # Display
        cv2.imshow('Debug Emoji Display', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Goodbye!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Debug demo completed!")

if __name__ == "__main__":
    print("🔍 Face Emoji Matcher - Debug Mode")
    print("=" * 50)
    
    # Test emoji display
    test_emoji_display()
    
    # Test similarity thresholds
    test_similarity_thresholds()
    
    # Run debug demo
    run_debug_demo()
