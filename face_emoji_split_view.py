#!/usr/bin/env python3
"""
Split-view Face Emoji Matcher - Shows your face on the left, matched emoji on the right.
"""

import cv2
import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV
import platform

class SplitViewFaceEmojiMatcher:
    def __init__(self):
        """Initialize the split-view face emoji matcher."""
        self.matcher = FaceEmojiMatcherOpenCV()
        self.emoji_size = 300  # Size of the emoji display
        self.padding = 20      # Padding between views
        
    def create_emoji_image(self, emoji: str, size: int = 300) -> np.ndarray:
        """Create an image with the emoji text."""
        # Create a white background
        emoji_img = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # Add emoji text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        color = (0, 0, 0)  # Black text
        thickness = 3
        
        # Get text size to center it
        text_size = cv2.getTextSize(emoji, font, font_scale, thickness)[0]
        text_x = (size - text_size[0]) // 2
        text_y = (size + text_size[1]) // 2
        
        # Draw the emoji
        cv2.putText(emoji_img, emoji, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add a border
        cv2.rectangle(emoji_img, (5, 5), (size-5, size-5), (0, 0, 0), 2)
        
        return emoji_img
    
    def create_similarity_bar(self, similarity: float, width: int = 300, height: int = 50) -> np.ndarray:
        """Create a similarity bar visualization."""
        # Create background
        bar_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Calculate bar width
        bar_width = int(width * similarity)
        
        # Choose color based on similarity
        if similarity > 0.7:
            color = (0, 255, 0)  # Green
        elif similarity > 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Draw the bar
        cv2.rectangle(bar_img, (0, 0), (bar_width, height), color, -1)
        
        # Add border
        cv2.rectangle(bar_img, (0, 0), (width, height), (0, 0, 0), 2)
        
        # Add text
        text = f"{similarity:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_color = (0, 0, 0)
        thickness = 2
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(bar_img, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        return bar_img
    
    def create_split_view(self, face_frame: np.ndarray, emoji: str, similarity: float) -> np.ndarray:
        """Create a split view with face on left and emoji on right."""
        # Resize face frame to a standard size
        face_height, face_width = 400, 400
        face_resized = cv2.resize(face_frame, (face_width, face_height))
        
        # Create emoji image
        emoji_img = self.create_emoji_image(emoji, self.emoji_size)
        
        # Create similarity bar
        similarity_bar = self.create_similarity_bar(similarity, self.emoji_size, 50)
        
        # Create the right panel (emoji + similarity)
        right_panel_height = self.emoji_size + self.padding + 50 + self.padding
        right_panel = np.ones((right_panel_height, self.emoji_size, 3), dtype=np.uint8) * 240
        
        # Place emoji at top
        right_panel[0:self.emoji_size, 0:self.emoji_size] = emoji_img
        
        # Place similarity bar below emoji
        bar_y = self.emoji_size + self.padding
        right_panel[bar_y:bar_y+50, 0:self.emoji_size] = similarity_bar
        
        # Add title text
        title_text = "Your Match!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 0)
        thickness = 2
        
        title_size = cv2.getTextSize(title_text, font, font_scale, thickness)[0]
        title_x = (self.emoji_size - title_size[0]) // 2
        title_y = 30
        
        cv2.putText(right_panel, title_text, (title_x, title_y), font, font_scale, color, thickness)
        
        # Create the combined view
        total_width = face_width + self.padding + self.emoji_size
        total_height = max(face_height, right_panel_height)
        
        combined = np.ones((total_height, total_width, 3), dtype=np.uint8) * 240
        
        # Place face on the left
        combined[0:face_height, 0:face_width] = face_resized
        
        # Place right panel on the right
        right_x = face_width + self.padding
        combined[0:right_panel_height, right_x:right_x+self.emoji_size] = right_panel
        
        # Add separator line
        cv2.line(combined, (face_width + self.padding//2, 0), 
                (face_width + self.padding//2, total_height), (0, 0, 0), 2)
        
        # Add title for face side
        face_title = "Your Face"
        face_title_size = cv2.getTextSize(face_title, font, font_scale, thickness)[0]
        face_title_x = (face_width - face_title_size[0]) // 2
        face_title_y = 30
        
        cv2.putText(combined, face_title, (face_title_x, face_title_y), font, font_scale, color, thickness)
        
        return combined
    
    def run_split_view_demo(self):
        """Run the split-view demo."""
        print("🎭 Face Emoji Matcher - Split View Demo")
        print("=" * 60)
        print("Instructions:")
        print("- Your face will appear on the LEFT")
        print("- Your matched emoji will appear on the RIGHT")
        print("- Make different facial expressions to see changes")
        print("- Press 'q' to quit, 's' to save current emoji")
        print("- Press 'h' to show this help")
        print("=" * 60)
        
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
            print("Make sure your camera is connected and not being used by another application.")
            return
        
        print("✅ Camera opened successfully!")
        print("Starting split-view face detection...")
        
        frame_count = 0
        last_emoji = "😐"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
            
            # Process the frame to get emoji match
            processed_frame, emoji, similarity = self.matcher.process_frame(frame)
            
            # Create split view
            split_view = self.create_split_view(processed_frame, emoji, similarity)
            
            # Update display every 10 frames to reduce flickering
            if frame_count % 10 == 0:
                if emoji != last_emoji:
                    print(f"🎭 New match: {emoji} (similarity: {similarity:.2f})")
                    last_emoji = emoji
            
            # Display the split view
            cv2.imshow('Face Emoji Matcher - Split View', split_view)
            
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
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("🎉 Split-view demo completed!")

if __name__ == "__main__":
    matcher = SplitViewFaceEmojiMatcher()
    matcher.run_split_view_demo()
