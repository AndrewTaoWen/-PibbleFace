# 🎭 Face Emoji Matcher

A computer vision project that uses facial feature analysis to match your face to the most similar emoji in real-time!

## 🌟 Features

- **Real-time face detection** using MediaPipe
- **Facial feature extraction** (smile intensity, eye openness, eyebrow position, mouth width)
- **Intelligent emoji matching** with 80+ emojis
- **Web interface** for easy interaction
- **Command-line demo** for quick testing
- **Similarity scoring** to show match confidence

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher (Python 3.11-3.12 recommended for full features)
- Webcam/camera access
- Internet connection (for initial package installation)

**Note for Python 3.13 users:** MediaPipe is not yet compatible with Python 3.13. Use the OpenCV version instead (see below).

### Installation

1. **Clone or download this project**
2. **Run the automated setup:**
   ```bash
   python3 setup.py
   ```
   
   This will:
   - Create a virtual environment called `venv`
   - Install all required dependencies
   - Run tests to verify everything works
   
   **Alternative manual setup:**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate     # On Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

### Running the Application

**First, activate the virtual environment:**
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Or use the provided scripts:
./activate.sh      # macOS/Linux
activate.bat       # Windows
```

#### Option 1: Web Interface (Recommended)
```bash
python app.py
```
Then open your browser and go to: `http://localhost:5000`

#### Option 2: Simple Command-Line Demo
```bash
python simple_demo.py
```

#### Option 3: OpenCV Version (Python 3.13 Compatible)
If you're using Python 3.13 or having issues with MediaPipe:
```bash
python simple_demo_opencv.py
```

#### Option 4: Split View (Face Left, Emoji Right)
Beautiful side-by-side view showing your face and matched emoji:
```bash
python face_emoji_split_view.py
```

#### Option 5: Enhanced View (With Feature Analysis)
Advanced view with facial feature analysis and similarity bars:
```bash
python face_emoji_enhanced_view.py
```

#### Option 6: Working Split View (FIXED - No More ???)
Fixed version that shows emoji names instead of symbols:
```bash
python face_emoji_working_split.py
```

#### Option 7: Simple Text Version
Basic version with emoji names as text:
```bash
python face_emoji_simple_text.py
```

#### Option 8: Mouth-Focused Version (FIXED - No More Screaming Default)
Prioritizes mouth expressions over eye/eyebrow features:
```bash
python face_emoji_mouth_focused.py
```

#### Option 9: Mouth-Focused Split View
Split view with mouth-focused weights:
```bash
python face_emoji_mouth_focused_split.py
```

#### Option 10: Simple Similarity Version (EQUAL WEIGHTS)
Just finds the most similar emoji with equal weights:
```bash
python face_emoji_simple_similarity.py
```

#### Option 11: Simple Similarity Split View
Split view with equal weights:
```bash
python face_emoji_simple_similarity_split.py
```

#### Option 12: Easy Launcher
Choose your preferred view:
```bash
python run_split_view.py
```

**To deactivate the virtual environment when done:**
```bash
deactivate
```

## 🎮 How to Use

### Web Interface
1. Allow camera access when prompted
2. Position your face in front of the camera
3. Make different facial expressions
4. Watch as the system matches your face to emojis!
5. Use the "Save Current Emoji" button to save matches
6. Click "Show All Emojis" to see all available emojis

### Command-Line Demo
- Press `q` to quit
- Press `s` to save current emoji
- Press `h` to show help

## 🔧 Technical Details

### Facial Features Analyzed
- **Smile Intensity**: How much you're smiling (0-1)
- **Eye Openness**: How open your eyes are (0-1)
- **Eyebrow Raise**: How raised your eyebrows are (0-1)
- **Mouth Width**: How wide your mouth is (0-1)

### Emoji Database
The system includes 80+ emojis with pre-defined facial characteristics:
- Happy faces: 😊, 😄, 😃, 😁, 😆, etc.
- Sad faces: 😭, 😢, 😔, 😕, etc.
- Surprised faces: 😮, 😯, 😲, 😱, etc.
- And many more!

### Matching Algorithm
The system uses weighted Euclidean distance to calculate similarity between your facial features and emoji characteristics, with weights:
- Smile intensity: 40%
- Eye openness: 30%
- Eyebrow raise: 20%
- Mouth width: 10%

## 📁 Project Structure

```
PibbleFace/
├── face_emoji_matcher.py           # Core face detection and emoji matching logic (MediaPipe)
├── face_emoji_matcher_opencv.py    # OpenCV version (Python 3.13 compatible)
├── face_emoji_split_view.py        # Split view (face left, emoji right)
├── face_emoji_enhanced_view.py     # Enhanced view with feature analysis
├── face_emoji_matcher_alternatives.py # Multiple detection methods
├── app.py                          # Flask web application
├── simple_demo.py                 # Command-line demo (MediaPipe)
├── simple_demo_opencv.py          # Command-line demo (OpenCV)
├── simple_demo_macos.py           # macOS optimized demo
├── run_split_view.py              # Easy launcher for different views
├── setup.py                       # Automated setup script
├── setup_python313.py             # Python 3.13 compatible setup
├── test_system.py                 # Test suite (MediaPipe)
├── test_opencv_version.py         # Test suite (OpenCV)
├── requirements.txt               # Python dependencies
├── activate.sh                    # Virtual environment activation (macOS/Linux)
├── activate.bat                   # Virtual environment activation (Windows)
├── venv/                          # Virtual environment (created by setup)
├── templates/
│   └── index.html                # Web interface
└── README.md                     # This file
```

## 🛠️ Troubleshooting

### Camera Issues
- Make sure your camera is not being used by another application
- Check camera permissions in your browser/system settings
- Try refreshing the page or restarting the application

### Installation Issues
- Make sure you have Python 3.7+ installed
- Try using `pip3` instead of `pip`
- On macOS, you might need to install Xcode command line tools: `xcode-select --install`

### Python 3.13 Compatibility Issues
- **MediaPipe not available**: Use the OpenCV version instead: `python simple_demo_opencv.py`
- **Distutils error**: This is expected with Python 3.13. The OpenCV version works around this.
- **Alternative**: Use Python 3.11 or 3.12 for the full MediaPipe experience

### Emoji Display Issues
- **Emojis showing as ???**: This is a common OpenCV issue. Use the fixed versions:
  - `python face_emoji_working_split.py` (recommended)
  - `python face_emoji_simple_text.py`
- **Why this happens**: OpenCV doesn't support Unicode emoji rendering well
- **Solution**: The fixed versions show emoji names (like "HAPPY", "SAD") instead of symbols

### Defaulting to Screaming Issue
- **Always showing screaming emoji**: This happens when eye/eyebrow features are weighted too heavily
- **Solution**: Use the mouth-focused or simple similarity versions:
  - `python face_emoji_mouth_focused.py` (mouth-focused)
  - `python face_emoji_mouth_focused_split.py` (mouth-focused split)
  - `python face_emoji_simple_similarity.py` (equal weights)
  - `python face_emoji_simple_similarity_split.py` (equal weights split)
- **Mouth-focused weights**: Smile 60%, Mouth Width 30%, Eye Openness 5%, Eyebrow Raise 5%
- **Equal weights**: All features 25% each

### Performance Issues
- Close other applications using the camera
- Ensure good lighting for better face detection
- Try adjusting the camera resolution in your system settings

## 🎯 Tips for Best Results

1. **Good Lighting**: Ensure your face is well-lit
2. **Clear View**: Keep your face clearly visible in the camera
3. **Stable Position**: Hold still for a moment to get accurate readings
4. **Varied Expressions**: Try different expressions to see various emoji matches
5. **Distance**: Stay about 1-2 feet from the camera

## 🔮 Future Enhancements

- [ ] Add more emoji categories
- [ ] Improve facial feature detection accuracy
- [ ] Add emotion recognition
- [ ] Support for multiple faces
- [ ] Mobile app version
- [ ] Custom emoji creation
- [ ] Social sharing features

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new emojis to the database
- Improving the matching algorithm
- Enhancing the user interface
- Adding new features

## 📧 Support

If you encounter any issues or have questions, please create an issue in the project repository.

---

**Have fun matching your face to emojis! 🎭✨**
