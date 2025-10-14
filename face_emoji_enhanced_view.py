#!/usr/bin/env python3
"""
Enhanced Face Emoji Matcher - Beautiful side-by-side view with detailed information.
"""

import cv2
import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV
import platform

class EnhancedFaceEmojiMatcher:
    def __init__(self):
        """Initialize the enhanced face emoji matcher."""
        self.matcher = FaceEmojiMatcherOpenCV()
        self.emoji_size = 400
        self.face_size = 500
        self.padding = 30
        
    def create_emoji_display(self, emoji: str, similarity: float, features: dict = None) -> np.ndarray:
        """Create a beautiful emoji display with information."""
        # Create the main emoji display
        emoji_display = np.ones((self.emoji_size, self.emoji_size, 3), dtype=np.uint8) * 250
        
        # Add gradient background
        for i in range(self.emoji_size):
            color_val = int(240 + (i / self.emoji_size) * 15)
            emoji_display[i, :] = [color_val, color_val, color_val]
        
        # Add emoji (large)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4.0
        color = (0, 0, 0)
        thickness = 4
        
        # Get text size to center it
        text_size = cv2.getTextSize(emoji, font, font_scale, thickness)[0]
        text_x = (self.emoji_size - text_size[0]) // 2
        text_y = (self.emoji_size + text_size[1]) // 2 - 50
        
        # Draw the emoji
        cv2.putText(emoji_display, emoji, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add similarity score
        similarity_text = f"Match: {similarity:.1%}"
        sim_font_scale = 1.2
        sim_color = (0, 100, 0) if similarity > 0.7 else (0, 100, 200) if similarity > 0.4 else (0, 0, 200)
        
        sim_text_size = cv2.getTextSize(similarity_text, font, sim_font_scale, 2)[0]
        sim_text_x = (self.emoji_size - sim_text_size[0]) // 2
        sim_text_y = text_y + 80
        
        cv2.putText(emoji_display, similarity_text, (sim_text_x, sim_text_y), 
                   font, sim_font_scale, sim_color, 2)
        
        # Add border with rounded corners effect
        cv2.rectangle(emoji_display, (10, 10), (self.emoji_size-10, self.emoji_size-10), (0, 0, 0), 3)
        
        return emoji_display
    
    def create_feature_bars(self, features: dict, width: int = 400, height: int = 200) -> np.ndarray:
        """Create feature visualization bars."""
        if not features:
            return np.ones((height, width, 3), dtype=np.uint8) * 240
        
        feature_img = np.ones((height, width, 3), dtype=np.uint8) * 250
        
        # Feature names and colors
        feature_info = [
            ("Smile", features.get('smile_intensity', 0), (0, 200, 0)),
            ("Eyes", features.get('eye_openness', 0), (0, 0, 200)),
            ("Eyebrows", features.get('eyebrow_raise', 0), (200, 0, 200)),
            ("Mouth", features.get('mouth_width', 0), (200, 100, 0))
        ]
        
        bar_height = 30
        bar_spacing = 40
        bar_width = width - 100
        
        for i, (name, value, color) in enumerate(feature_info):
            y_pos = 20 + i * bar_spacing
            
            # Draw background bar
            cv2.rectangle(feature_img, (50, y_pos), (50 + bar_width, y_pos + bar_height), (200, 200, 200), -1)
            
            # Draw value bar
            value_width = int(bar_width * value)
            cv2.rectangle(feature_img, (50, y_pos), (50 + value_width, y_pos + bar_height), color, -1)
            
            # Draw border
            cv2.rectangle(feature_img, (50, y_pos), (50 + bar_width, y_pos + bar_height), (0, 0, 0), 1)
            
            # Add label
            cv2.putText(feature_img, name, (5, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Add value
            value_text = f"{value:.2f}"
            cv2.putText(feature_img, value_text, (50 + bar_width + 10, y_pos + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add title
        cv2.putText(feature_img, "Facial Features", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return feature_img
    
    def create_enhanced_view(self, face_frame: np.ndarray, emoji: str, similarity: float, features: dict = None) -> np.ndarray:
        """Create the enhanced split view."""
        # Resize face frame
        face_resized = cv2.resize(face_frame, (self.face_size, self.face_size))
        
        # Create emoji display
        emoji_display = self.create_emoji_display(emoji, similarity, features)
        
        # Create feature bars
        feature_bars = self.create_feature_bars(features, self.emoji_size, 200)
        
        # Create the right panel
        right_panel_height = self.emoji_size + self.padding + 200
        right_panel = np.ones((right_panel_height, self.emoji_size, 3), dtype=np.uint8) * 240
        
        # Place emoji display at top
        right_panel[0:self.emoji_size, 0:self.emoji_size] = emoji_display
        
        # Place feature bars below
        feature_y = self.emoji_size + self.padding
        right_panel[feature_y:feature_y+200, 0:self.emoji_size] = feature_bars
        
        # Create the combined view
        total_width = self.face_size + self.padding + self.emoji_size
        total_height = max(self.face_size, right_panel_height)
        
        combined = np.ones((total_height, total_width, 3), dtype=np.uint8) * 240
        
        # Place face on the left
        combined[0:self.face_size, 0:self.face_size] = face_resized
        
        # Place right panel on the right
        right_x = self.face_size + self.padding
        combined[0:right_panel_height, right_x:right_x+self.emoji_size] = right_panel
        
        # Add separator line
        cv2.line(combined, (self.face_size + self.padding//2, 0), 
                (self.face_size + self.padding//2, total_height), (100, 100, 100), 3)
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 0)
        thickness = 2
        
        # Face title
        face_title = "Your Face"
        face_title_size = cv2.getTextSize(face_title, font, font_scale, thickness)[0]
        face_title_x = (self.face_size - face_title_size[0]) // 2
        face_title_y = 30
        
        cv2.putText(combined, face_title, (face_title_x, face_title_y), font, font_scale, color, thickness)
        
        # Emoji title
        emoji_title = "Your Emoji Match"
        emoji_title_size = cv2.getTextSize(emoji_title, font, font_scale, thickness)[0]
        emoji_title_x = right_x + (self.emoji_size - emoji_title_size[0]) // 2
        emoji_title_y = 30
        
        cv2.putText(combined, emoji_title, (emoji_title_x, emoji_title_y), font, font_scale, color, thickness)
        
        return combined
    
    def run_enhanced_demo(self):
        """Run the enhanced demo."""
        print("🎭 Face Emoji Matcher - Enhanced Split View")
        print("=" * 60)
        print("Features:")
        print("- Your face on the LEFT")
        print("- Matched emoji on the RIGHT")
        print("- Real-time facial feature analysis")
        print("- Similarity scoring")
        print("- Press 'q' to quit, 's' to save, 'h' for help")
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
            return
        
        print("✅ Starting enhanced face detection...")
        
        frame_count = 0
        last_emoji = "😐"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame, emoji, similarity = self.matcher.process_frame(frame)
            
            # Extract features for display
            features = self.matcher.extract_facial_features(frame)
            
            # Create enhanced view
            enhanced_view = self.create_enhanced_view(processed_frame, emoji, similarity, features)
            
            # Update display
            if frame_count % 10 == 0:
                if emoji != last_emoji:
                    print(f"🎭 New match: {emoji} (similarity: {similarity:.2f})")
                    last_emoji = emoji
            
            # Display
            cv2.imshow('Face Emoji Matcher - Enhanced View', enhanced_view)
            
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
                print("- 'q': Quit")
                print("- 's': Save current match")
                print("- 'h': Show help")
                print("=" * 60)
        
        cap.release()
        cv2.destroyAllWindows()
        print("🎉 Enhanced demo completed!")

if __name__ == "__main__":
    matcher = EnhancedFaceEmojiMatcher()
    matcher.run_enhanced_demo()
