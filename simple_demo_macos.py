#!/usr/bin/env python3
"""
macOS-optimized demo of the Face Emoji Matcher.
This version handles macOS camera permissions and device access better.
"""

import cv2
import numpy as np
import platform
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

def check_camera_permissions():
    """Check and provide guidance on camera permissions."""
    print("🔍 Checking camera setup...")
    
    if platform.system() == "Darwin":  # macOS
        print("📱 macOS detected")
        print("If you get camera permission errors:")
        print("1. Go to System Preferences > Security & Privacy > Camera")
        print("2. Make sure Terminal (or your IDE) has camera access")
        print("3. Or run this from Terminal.app instead of your IDE")
        print()
    
    return True

def find_working_camera():
    """Find a working camera by trying different indices."""
    print("🔍 Searching for available cameras...")
    
    # Try different camera indices
    camera_indices = [0, 1, 2, -1]  # -1 for default camera
    
    for camera_idx in camera_indices:
        print(f"🔄 Trying camera index {camera_idx}...")
        cap = cv2.VideoCapture(camera_idx)
        
        if cap.isOpened():
            # Try to read a frame to make sure it's working
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {camera_idx} is working!")
                cap.release()
                return camera_idx
            else:
                print(f"❌ Camera {camera_idx} opened but can't read frames")
        else:
            print(f"❌ Camera {camera_idx} failed to open")
        
        cap.release()
    
    return None

def main():
    print("🎭 Face Emoji Matcher - macOS Optimized Demo")
    print("=" * 60)
    
    # Check camera permissions
    check_camera_permissions()
    
    # Find working camera
    camera_idx = find_working_camera()
    
    if camera_idx is None:
        print("❌ No working camera found!")
        print("\nTroubleshooting:")
        print("1. Make sure your camera is connected")
        print("2. Check camera permissions in System Preferences")
        print("3. Close other applications using the camera")
        print("4. Try running from Terminal.app instead of your IDE")
        return
    
    print(f"✅ Using camera {camera_idx}")
    print("\nInstructions:")
    print("- Position your face in front of the camera")
    print("- Make different facial expressions")
    print("- Press 'q' to quit")
    print("- Press 's' to save current emoji")
    print("- Press 'h' to show this help")
    print("=" * 60)
    
    # Initialize the matcher
    matcher = FaceEmojiMatcherOpenCV()
    
    # Open the working camera
    cap = cv2.VideoCapture(camera_idx)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
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
        cv2.imshow('Face Emoji Matcher - macOS Demo', processed_frame)
        
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
