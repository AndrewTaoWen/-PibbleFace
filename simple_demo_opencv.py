#!/usr/bin/env python3
"""
Simple command-line demo of the Face Emoji Matcher using OpenCV.
This version works without MediaPipe and is compatible with Python 3.13.
"""

import cv2
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

def main():
    print("🎭 Face Emoji Matcher - OpenCV Demo (Python 3.13 Compatible)")
    print("=" * 60)
    print("Instructions:")
    print("- Position your face in front of the camera")
    print("- Make different facial expressions")
    print("- Press 'q' to quit")
    print("- Press 's' to save current emoji")
    print("- Press 'h' to show this help")
    print("=" * 60)
    
    # Initialize the matcher
    matcher = FaceEmojiMatcherOpenCV()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("Make sure your camera is connected and not being used by another application.")
        return
    
    print("✅ Camera opened successfully!")
    print("Starting face detection...")
    
    frame_count = 0
    last_emoji = "😐"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame from camera")
            break
        
        # Process the frame
        processed_frame, emoji, similarity = matcher.process_frame(frame)
        
        # Update display every 10 frames to reduce flickering
        if frame_count % 10 == 0:
            if emoji != last_emoji:
                print(f"🎭 New match: {emoji} (similarity: {similarity:.2f})")
                last_emoji = emoji
        
        # Display the frame
        cv2.imshow('Face Emoji Matcher - OpenCV Demo', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Goodbye!")
            break
        elif key == ord('s'):
            print(f"💾 Saved: {emoji} (similarity: {similarity:.2f})")
        elif key == ord('h'):
            print("\n" + "=" * 60)
            print("Help:")
            print("- 'q': Quit the application")
            print("- 's': Save current emoji match")
            print("- 'h': Show this help message")
            print("=" * 60)
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Demo completed!")

if __name__ == "__main__":
    main()
