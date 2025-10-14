#!/usr/bin/env python3
"""
Debug version to see what facial features are being detected in real-time.
"""

import cv2
import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

def main():
    print("🔍 Face Detection Debug Tool")
    print("=" * 50)
    print("This will show you exactly what features are being detected")
    print("Press 'q' to quit")
    print("=" * 50)
    
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
    
    print("✅ Starting debug detection...")
    print("Watch the console for feature values!")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract features
        features = matcher.extract_facial_features(frame)
        
        # Process frame for display
        processed_frame, emoji, similarity = matcher.process_frame(frame)
        
        # Create debug display
        debug_frame = processed_frame.copy()
        
        # Add feature information to the frame
        if features:
            y_pos = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (0, 255, 0)
            thickness = 2
            
            cv2.putText(debug_frame, f"Smile: {features['smile_intensity']:.2f}", (10, y_pos), font, font_scale, color, thickness)
            y_pos += 30
            cv2.putText(debug_frame, f"Eyes: {features['eye_openness']:.2f}", (10, y_pos), font, font_scale, color, thickness)
            y_pos += 30
            cv2.putText(debug_frame, f"Eyebrows: {features['eyebrow_raise']:.2f}", (10, y_pos), font, font_scale, color, thickness)
            y_pos += 30
            cv2.putText(debug_frame, f"Mouth: {features['mouth_width']:.2f}", (10, y_pos), font, font_scale, color, thickness)
            y_pos += 30
            cv2.putText(debug_frame, f"Match: {emoji} ({similarity:.3f})", (10, y_pos), font, font_scale, color, thickness)
        else:
            cv2.putText(debug_frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_frame, f"Default: {emoji} ({similarity:.3f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Print to console every 30 frames
        if frame_count % 30 == 0:
            if features:
                print(f"Frame {frame_count}: Smile={features['smile_intensity']:.2f}, Eyes={features['eye_openness']:.2f}, Eyebrows={features['eyebrow_raise']:.2f}, Mouth={features['mouth_width']:.2f} -> {emoji} ({similarity:.3f})")
            else:
                print(f"Frame {frame_count}: NO FACE DETECTED -> {emoji} ({similarity:.3f})")
        
        # Display
        cv2.imshow('Face Detection Debug', debug_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Goodbye!")
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎉 Debug completed!")

if __name__ == "__main__":
    main()
