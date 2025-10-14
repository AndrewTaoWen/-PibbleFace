#!/usr/bin/env python3
"""
Fixed Face Emoji Matcher - Uses emoji names instead of Unicode symbols for better compatibility.
"""

import cv2
import numpy as np
from face_emoji_matcher_opencv import FaceEmojiMatcherOpenCV

class FixedFaceEmojiMatcher:
    def __init__(self):
        """Initialize the fixed face emoji matcher."""
        self.matcher = FaceEmojiMatcherOpenCV()
        
        # Create a mapping of emojis to readable names
        self.emoji_names = {
            '😊': 'Happy',
            '😄': 'Very Happy',
            '😃': 'Excited',
            '😁': 'Grinning',
            '😆': 'Laughing',
            '😅': 'Sweat Smile',
            '🤣': 'ROFL',
            '😂': 'Tears of Joy',
            '😭': 'Crying',
            '😢': 'Sad',
            '😔': 'Pensive',
            '😕': 'Confused',
            '😐': 'Neutral',
            '😑': 'Expressionless',
            '😒': 'Unamused',
            '🙄': 'Eye Roll',
            '😏': 'Smirk',
            '😎': 'Cool',
            '🤔': 'Thinking',
            '😮': 'Surprised',
            '😯': 'Hushed',
            '😲': 'Astonished',
            '😳': 'Flushed',
            '😱': 'Screaming',
            '😨': 'Fearful',
            '😰': 'Anxious',
            '😴': 'Sleeping',
            '😪': 'Sleepy',
            '🤤': 'Drooling',
            '😋': 'Yummy',
            '😍': 'Heart Eyes',
            '🥰': 'Smiling Hearts',
            '😘': 'Kiss',
            '😗': 'Kissing',
            '😙': 'Kissing Smile',
            '😚': 'Kissing Closed Eyes',
            '🤗': 'Hugging',
            '🤩': 'Star Eyes',
            '🤨': 'Raised Eyebrow',
            '🧐': 'Monocle',
            '🤓': 'Nerd',
            '😤': 'Huffing',
            '😠': 'Angry',
            '😡': 'Raging',
            '🤬': 'Cursing',
            '🤯': 'Exploding Head',
            '😵': 'Dizzy',
            '🥴': 'Woozy',
            '😵‍💫': 'Spiral Eyes',
            '🤢': 'Nauseated',
            '🤮': 'Vomiting',
            '🤧': 'Sneezing',
            '🥵': 'Hot',
            '🥶': 'Cold',
            '😶': 'No Mouth',
            '😶‍🌫️': 'Cloud',
            '😷': 'Mask',
            '🤒': 'Sick',
            '🤕': 'Injured',
            '🤑': 'Money Mouth',
            '🤠': 'Cowboy',
            '🥳': 'Party',
            '🥸': 'Disguise',
            '😈': 'Smiling Devil',
            '👿': 'Angry Devil',
            '👹': 'Ogre',
            '👺': 'Goblin',
            '💀': 'Skull',
            '☠️': 'Skull Crossbones',
            '💩': 'Poop',
            '🤡': 'Clown',
            '👻': 'Ghost',
            '👽': 'Alien',
            '👾': 'Alien Monster',
            '🤖': 'Robot',
            '😺': 'Smiling Cat',
            '😸': 'Grinning Cat',
            '😹': 'Joy Cat',
            '😻': 'Heart Eyes Cat',
            '😼': 'Smirk Cat',
            '😽': 'Kissing Cat',
            '🙀': 'Scream Cat',
            '😿': 'Crying Cat',
            '😾': 'Pouting Cat'
        }
    
    def create_emoji_display(self, emoji: str, similarity: float, size: int = 300) -> np.ndarray:
        """Create an emoji display using text names instead of Unicode."""
        # Create background
        emoji_img = np.ones((size, size, 3), dtype=np.uint8) * 250
        
        # Add gradient background
        for i in range(size):
            color_val = int(240 + (i / size) * 15)
            emoji_img[i, :] = [color_val, color_val, color_val]
        
        # Get emoji name
        emoji_name = self.emoji_names.get(emoji, "Unknown")
        
        # Add emoji name (large)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        color = (0, 0, 0)
        thickness = 2
        
        # Split long names into multiple lines
        words = emoji_name.split()
        if len(words) > 2:
            line1 = " ".join(words[:len(words)//2])
            line2 = " ".join(words[len(words)//2:])
            lines = [line1, line2]
        else:
            lines = [emoji_name]
        
        # Draw each line
        line_height = 40
        start_y = size // 2 - (len(lines) * line_height) // 2
        
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = (size - text_size[0]) // 2
            text_y = start_y + i * line_height
            cv2.putText(emoji_img, line, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add similarity score
        similarity_text = f"Match: {similarity:.1%}"
        sim_font_scale = 0.8
        sim_color = (0, 100, 0) if similarity > 0.7 else (0, 100, 200) if similarity > 0.4 else (0, 0, 200)
        
        sim_text_size = cv2.getTextSize(similarity_text, font, sim_font_scale, 2)[0]
        sim_text_x = (size - sim_text_size[0]) // 2
        sim_text_y = start_y + len(lines) * line_height + 30
        
        cv2.putText(emoji_img, similarity_text, (sim_text_x, sim_text_y), 
                   font, sim_font_scale, sim_color, 2)
        
        # Add border
        cv2.rectangle(emoji_img, (10, 10), (size-10, size-10), (0, 0, 0), 3)
        
        return emoji_img
    
    def create_split_view(self, face_frame: np.ndarray, emoji: str, similarity: float) -> np.ndarray:
        """Create a split view with face on left and emoji info on right."""
        # Resize face frame
        face_height, face_width = 400, 400
        face_resized = cv2.resize(face_frame, (face_width, face_height))
        
        # Create emoji display
        emoji_display = self.create_emoji_display(emoji, similarity, 300)
        
        # Create the combined view
        total_width = face_width + 30 + 300
        total_height = max(face_height, 300)
        
        combined = np.ones((total_height, total_width, 3), dtype=np.uint8) * 240
        
        # Place face on the left
        combined[0:face_height, 0:face_width] = face_resized
        
        # Place emoji display on the right
        right_x = face_width + 30
        combined[0:300, right_x:right_x+300] = emoji_display
        
        # Add separator line
        cv2.line(combined, (face_width + 15, 0), 
                (face_width + 15, total_height), (100, 100, 100), 3)
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 0)
        thickness = 2
        
        # Face title
        face_title = "Your Face"
        face_title_size = cv2.getTextSize(face_title, font, font_scale, thickness)[0]
        face_title_x = (face_width - face_title_size[0]) // 2
        face_title_y = 30
        
        cv2.putText(combined, face_title, (face_title_x, face_title_y), font, font_scale, color, thickness)
        
        # Emoji title
        emoji_title = "Your Match"
        emoji_title_size = cv2.getTextSize(emoji_title, font, font_scale, thickness)[0]
        emoji_title_x = right_x + (300 - emoji_title_size[0]) // 2
        emoji_title_y = 30
        
        cv2.putText(combined, emoji_title, (emoji_title_x, emoji_title_y), font, font_scale, color, thickness)
        
        return combined
    
    def run_fixed_demo(self):
        """Run the fixed demo."""
        print("🎭 Face Emoji Matcher - Fixed Display")
        print("=" * 50)
        print("Features:")
        print("- Your face on the LEFT")
        print("- Emoji name and match score on the RIGHT")
        print("- No more ??? symbols!")
        print("- Press 'q' to quit, 's' to save")
        print("=" * 50)
        
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
        
        print("✅ Starting fixed face detection...")
        
        frame_count = 0
        last_emoji = "😐"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame, emoji, similarity = self.matcher.process_frame(frame)
            
            # Create split view
            split_view = self.create_split_view(processed_frame, emoji, similarity)
            
            # Update display
            if frame_count % 10 == 0:
                if emoji != last_emoji:
                    emoji_name = self.emoji_names.get(emoji, "Unknown")
                    print(f"🎭 New match: {emoji_name} ({emoji}) - similarity: {similarity:.3f}")
                    last_emoji = emoji
            
            # Display
            cv2.imshow('Face Emoji Matcher - Fixed Display', split_view)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Goodbye!")
                break
            elif key == ord('s'):
                emoji_name = self.emoji_names.get(emoji, "Unknown")
                print(f"💾 Saved: {emoji_name} ({emoji}) - similarity: {similarity:.3f}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("🎉 Fixed demo completed!")

if __name__ == "__main__":
    matcher = FixedFaceEmojiMatcher()
    matcher.run_fixed_demo()
