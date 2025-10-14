#!/usr/bin/env python3
"""
Mouth-focused Split-View Face Emoji Matcher - Shows face on left, emoji name on right.
Prioritizes mouth expressions over eye/eyebrow features.
"""

import cv2
import numpy as np
from face_emoji_mouth_focused import MouthFocusedFaceEmojiMatcher

class MouthFocusedSplitViewMatcher(MouthFocusedFaceEmojiMatcher):
    def __init__(self):
        """Initialize the mouth-focused split-view matcher."""
        super().__init__()
        
        # Emoji to name mapping
        self.emoji_names = {
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
    
    def create_emoji_panel(self, emoji: str, similarity: float, width: int = 400, height: int = 500) -> np.ndarray:
        """Create the right panel with emoji information."""
        # Create background
        panel = np.ones((height, width, 3), dtype=np.uint8) * 250
        
        # Add gradient background
        for i in range(height):
            color_val = int(240 + (i / height) * 15)
            panel[i, :] = [color_val, color_val, color_val]
        
        # Get emoji name
        emoji_name = self.emoji_names.get(emoji, "UNKNOWN")
        
        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        color = (0, 0, 0)
        thickness = 2
        
        title = "YOUR MATCH"
        title_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        title_x = (width - title_size[0]) // 2
        title_y = 50
        
        cv2.putText(panel, title, (title_x, title_y), font, font_scale, color, thickness)
        
        # Add emoji name (large)
        emoji_font_scale = 2.0
        emoji_color = (0, 100, 0)  # Dark green
        
        # Split long names into multiple lines
        words = emoji_name.split()
        if len(words) > 2:
            line1 = " ".join(words[:len(words)//2])
            line2 = " ".join(words[len(words)//2:])
            lines = [line1, line2]
        else:
            lines = [emoji_name]
        
        # Draw emoji name
        line_height = 60
        start_y = height // 2 - (len(lines) * line_height) // 2
        
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, emoji_font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = start_y + i * line_height
            cv2.putText(panel, line, (text_x, text_y), font, emoji_font_scale, emoji_color, thickness)
        
        # Add similarity bar
        bar_width = width - 100
        bar_height = 40
        bar_x = 50
        bar_y = start_y + len(lines) * line_height + 50
        
        # Draw bar background
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
        
        # Draw similarity fill
        fill_width = int(bar_width * similarity)
        bar_color = (0, 200, 0) if similarity > 0.7 else (0, 200, 200) if similarity > 0.4 else (0, 0, 200)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)
        
        # Draw bar border
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)
        
        # Add similarity text
        similarity_text = f"{similarity:.1%}"
        sim_text_size = cv2.getTextSize(similarity_text, font, 1.0, 2)[0]
        sim_text_x = (width - sim_text_size[0]) // 2
        sim_text_y = bar_y + bar_height + 40
        
        cv2.putText(panel, similarity_text, (sim_text_x, sim_text_y), font, 1.0, (0, 0, 0), 2)
        
        # Add weight info
        weight_text = "MOUTH FOCUSED"
        weight_text_size = cv2.getTextSize(weight_text, font, 0.6, 1)[0]
        weight_text_x = (width - weight_text_size[0]) // 2
        weight_text_y = sim_text_y + 30
        
        cv2.putText(panel, weight_text, (weight_text_x, weight_text_y), font, 0.6, (0, 100, 100), 1)
        
        # Add border
        cv2.rectangle(panel, (10, 10), (width-10, height-10), (0, 0, 0), 3)
        
        return panel
    
    def create_split_view(self, face_frame: np.ndarray, emoji: str, similarity: float) -> np.ndarray:
        """Create the split view."""
        # Resize face frame
        face_size = 500
        face_resized = cv2.resize(face_frame, (face_size, face_size))
        
        # Create emoji panel
        emoji_panel = self.create_emoji_panel(emoji, similarity, 400, 500)
        
        # Create combined view
        total_width = face_size + 30 + 400
        total_height = 500
        
        combined = np.ones((total_height, total_width, 3), dtype=np.uint8) * 240
        
        # Place face on the left
        combined[0:face_size, 0:face_size] = face_resized
        
        # Place emoji panel on the right
        right_x = face_size + 30
        combined[0:500, right_x:right_x+400] = emoji_panel
        
        # Add separator line
        cv2.line(combined, (face_size + 15, 0), (face_size + 15, total_height), (100, 100, 100), 3)
        
        # Add face title
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 0)
        thickness = 2
        
        face_title = "YOUR FACE"
        face_title_size = cv2.getTextSize(face_title, font, font_scale, thickness)[0]
        face_title_x = (face_size - face_title_size[0]) // 2
        face_title_y = 30
        
        cv2.putText(combined, face_title, (face_title_x, face_title_y), font, font_scale, color, thickness)
        
        return combined
    
    def run_mouth_focused_split_demo(self):
        """Run the mouth-focused split demo."""
        print("🎭 Face Emoji Matcher - Mouth Focused Split View")
        print("=" * 60)
        print("✅ NEW WEIGHTS:")
        print("   - Smile: 60% (was 40%)")
        print("   - Mouth Width: 30% (was 10%)")
        print("   - Eye Openness: 5% (was 30%)")
        print("   - Eyebrow Raise: 5% (was 20%)")
        print("=" * 60)
        print("✅ Your face on the LEFT")
        print("✅ Emoji name on the RIGHT")
        print("✅ No more defaulting to screaming!")
        print("✅ Press 'q' to quit, 's' to save")
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
        
        print("✅ Starting mouth-focused split detection...")
        
        frame_count = 0
        last_emoji = "😐"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame, emoji, similarity = self.process_frame(frame)
            
            # Create split view
            split_view = self.create_split_view(processed_frame, emoji, similarity)
            
            # Update console output
            if frame_count % 10 == 0:
                if emoji != last_emoji:
                    emoji_name = self.emoji_names.get(emoji, "UNKNOWN")
                    print(f"🎭 New match: {emoji_name} ({emoji}) - similarity: {similarity:.3f}")
                    last_emoji = emoji
            
            # Display
            cv2.imshow('Face Emoji Matcher - Mouth Focused Split', split_view)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Goodbye!")
                break
            elif key == ord('s'):
                emoji_name = self.emoji_names.get(emoji, "UNKNOWN")
                print(f"💾 Saved: {emoji_name} ({emoji}) - similarity: {similarity:.3f}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("🎉 Mouth-focused split demo completed!")

if __name__ == "__main__":
    matcher = MouthFocusedSplitViewMatcher()
    matcher.run_mouth_focused_split_demo()
