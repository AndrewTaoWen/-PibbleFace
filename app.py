from flask import Flask, render_template, Response, jsonify
import cv2
import base64
import numpy as np
from face_emoji_matcher import FaceEmojiMatcher
import threading
import time

app = Flask(__name__)
# Keep untrusted hostnames from reaching a local webcam through DNS rebinding.
app.config["TRUSTED_HOSTS"] = ["localhost", "127.0.0.1"]

@app.after_request
def protect_camera_response(response):
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


# Global variables for video streaming
matcher = FaceEmojiMatcher()
current_frame = None
current_emoji = "😐"
current_similarity = 0.0
frame_lock = threading.Lock()

def generate_frames():
    """Generate video frames for streaming."""
    global current_frame, current_emoji, current_similarity
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, emoji, similarity = matcher.process_frame(frame)
        
        # Update global variables
        with frame_lock:
            current_frame = processed_frame
            current_emoji = emoji
            current_similarity = similarity
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emoji')
def get_emoji():
    """Get current emoji and similarity."""
    with frame_lock:
        return jsonify({
            'emoji': current_emoji,
            'similarity': current_similarity
        })

@app.route('/get_all_emojis')
def get_all_emojis():
    """Get all available emojis."""
    return jsonify(list(matcher.emoji_database.keys()))

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
