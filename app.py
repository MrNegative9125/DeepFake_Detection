
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = r"Models\ResNet50Deepfake_best.pth"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# GPU/CPU DEVICE SETUP (FROM STREAMLIT CODE)
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # GPU Optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    print("✅ GPU optimizations enabled")
else:
    print("⚠️  Running on CPU (slower)")

# ============================================================================
# MODEL ARCHITECTURE (EXACT FROM STREAMLIT CODE)
# ============================================================================
class ResNet50Deepfake(nn.Module):
    """ResNet architecture tuned for artifact detection"""
    
    def __init__(self, dropout=0.6):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 3, dropout * 0.3)
        self.layer2 = self._make_layer(64, 128, 4, dropout * 0.4)
        self.layer3 = self._make_layer(128, 256, 6, dropout * 0.5)
        self.layer4 = self._make_layer(256, 512, 3, dropout * 0.6)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def _make_layer(self, in_ch, out_ch, blocks, dropout):
        layers = []
        for i in range(blocks):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
            if i == blocks - 1:
                layers.append(nn.MaxPool2d(2))
                layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

# ============================================================================
# MODEL LOADING (FROM STREAMLIT CODE)
# ============================================================================
def load_model(model_path, dropout=0.5):
    """Load the model"""
    print(f"\n📥 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    
    try:
        model = ResNet50Deepfake(dropout=dropout).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'val_acc' in checkpoint:
                    print(f"📊 Val Accuracy: {checkpoint['val_acc']:.4f}")
                if 'val_f1' in checkpoint:
                    print(f"📊 Val F1-Score: {checkpoint['val_f1']:.4f}")
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✅ Model loaded successfully\n")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

# Load model at startup
model = load_model(MODEL_PATH, dropout=0.5)

# ============================================================================
# PREPROCESSING (FROM STREAMLIT CODE)
# ============================================================================
def get_transform(img_size=224):
    """Get preprocessing transform"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

transform = get_transform()

# ============================================================================
# FACE DETECTION (FROM STREAMLIT CODE)
# ============================================================================
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✅ Face detector initialized\n")

def extract_face_with_context(frame, expand_ratio=1.5):
    """Extract face with surrounding region"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    if len(faces) == 0:
        return None
    
    # Get largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    
    # Expand region
    center_x, center_y = x + w // 2, y + h // 2
    new_w = int(w * expand_ratio)
    new_h = int(h * expand_ratio)
    
    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(frame.shape[1], center_x + new_w // 2)
    y2 = min(frame.shape[0], center_y + new_h // 2)
    
    face_region = frame[y1:y2, x1:x2]
    
    return face_region if face_region.size > 0 else None

# ============================================================================
# VIDEO PROCESSING (FROM STREAMLIT CODE)
# ============================================================================
def extract_all_frames(video_path, sample_rate=1, face_expansion_ratio=1.5):
    """Extract all frames (or every Nth frame) from video with face detection"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n🎬 Video info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Sample rate: every {sample_rate} frame(s)")
    
    frames = []
    frame_numbers = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            # Extract face
            face = extract_face_with_context(frame, expand_ratio=face_expansion_ratio)
            if face is not None:
                frames.append(face)
                frame_numbers.append(frame_idx)
        
        frame_idx += 1
        
        # Progress update
        if frame_idx % 100 == 0:
            print(f"   Processing frame {frame_idx}/{total_frames}, found {len(frames)} faces")
    
    cap.release()
    print(f"✅ Extraction complete: {len(frames)} frames with faces\n")
    
    return frames, frame_numbers, fps, total_frames

# ============================================================================
# INFERENCE (FROM STREAMLIT CODE)
# ============================================================================
def predict_frames(frames):
    """Run inference on all frames"""
    predictions = []
    probabilities = []
    
    print(f"🧠 Running inference on {len(frames)} frames...")
    
    with torch.no_grad():
        for idx, frame in enumerate(frames):
            # Preprocess
            transformed = transform(image=frame)
            img_tensor = transformed['image'].unsqueeze(0).to(device)
            
            # Predict
            output = model(img_tensor).squeeze()
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
            
            predictions.append(pred)
            probabilities.append(prob)
            
            # Progress update
            if idx % 50 == 0:
                print(f"   Analyzed {idx + 1}/{len(frames)} frames")
    
    print(f"✅ Inference complete!\n")
    
    return predictions, probabilities

# ============================================================================
# VIDEO ANALYSIS FUNCTION
# ============================================================================
def analyze_video(video_path, sample_rate=5, face_expansion=1.5):
    """Analyze video for deepfake detection"""
    print(f"\n{'='*70}")
    print(f"🎬 ANALYZING VIDEO")
    print(f"{'='*70}")
    print(f"📂 Video: {video_path}")
    
    # Extract frames
    frames, frame_numbers, fps, total_frames = extract_all_frames(
        video_path, 
        sample_rate=sample_rate,
        face_expansion_ratio=face_expansion
    )
    
    if len(frames) == 0:
        return {
            'error': 'No faces detected in video',
            'prediction': None,
            'confidence': None
        }
    
    # Run predictions
    predictions, probabilities = predict_frames(frames)
    
    # Calculate statistics
    avg_prob = np.mean(probabilities)
    fake_frames = sum(predictions)
    fake_percentage = (fake_frames / len(predictions)) * 100
    
    prediction = "FAKE" if avg_prob > 0.5 else "REAL"
    confidence = max(avg_prob, 1 - avg_prob)
    
    print(f"{'='*70}")
    print(f"📊 RESULTS")
    print(f"{'='*70}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Average Probability: {avg_prob:.4f}")
    print(f"Fake Frames: {fake_frames}/{len(predictions)} ({fake_percentage:.1f}%)")
    print(f"{'='*70}\n")
    
    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'frames_analyzed': len(frames),
        'total_frames': total_frames,
        'fake_probability': float(avg_prob),
        'real_probability': float(1 - avg_prob),
        'fake_frames_count': fake_frames,
        'fake_frames_percentage': float(fake_percentage),
        'probabilities': probabilities,
        'frame_numbers': frame_numbers,
        'fps': fps
    }

# ============================================================================
# CAMERA CAPTURE GLOBALS
# ============================================================================
camera_frames = []
camera_lock = threading.Lock()
is_capturing = False

# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and analysis"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Analyze video (sample every 5th frame by default)
            result = analyze_video(filepath, sample_rate=5, face_expansion=1.5)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(result)
        
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start camera capture"""
    global camera_frames, is_capturing
    
    with camera_lock:
        camera_frames = []
        is_capturing = True
    
    print("\n📹 Camera capture started")
    return jsonify({'status': 'Camera started', 'target_frames': 300})

@app.route('/camera/frame', methods=['POST'])
def receive_frame():
    """Receive frame from camera"""
    global camera_frames, is_capturing
    
    if not is_capturing:
        return jsonify({'error': 'Camera not started'}), 400
    
    data = request.json
    img_data = data.get('frame')
    
    if not img_data:
        return jsonify({'error': 'No frame data'}), 400
    
    try:
        # Decode base64 image
        img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Extract face
        face = extract_face_with_context(frame, expand_ratio=1.5)
        
        if face is not None:
            with camera_lock:
                if len(camera_frames) < 300:
                    camera_frames.append(face)
                    frames_count = len(camera_frames)
                else:
                    frames_count = 300
            
            return jsonify({
                'status': 'Frame received',
                'frames_collected': frames_count,
                'target': 300,
                'face_detected': True
            })
        else:
            return jsonify({
                'status': 'No face detected',
                'frames_collected': len(camera_frames),
                'target': 300,
                'face_detected': False
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera/predict', methods=['POST'])
def predict_camera():
    """Predict from camera frames"""
    global camera_frames, is_capturing
    
    with camera_lock:
        is_capturing = False
        frames = camera_frames.copy()
        camera_frames = []
    
    if len(frames) == 0:
        return jsonify({'error': 'No frames collected'}), 400
    
    print(f"\n📹 Analyzing {len(frames)} camera frames...")
    
    # Run predictions
    predictions, probabilities = predict_frames(frames)
    
    # Calculate statistics
    avg_prob = np.mean(probabilities)
    fake_frames = sum(predictions)
    fake_percentage = (fake_frames / len(predictions)) * 100
    
    prediction = "FAKE" if avg_prob > 0.5 else "REAL"
    confidence = max(avg_prob, 1 - avg_prob)
    
    print(f"✅ Camera analysis complete: {prediction} ({confidence*100:.2f}%)\n")
    
    return jsonify({
        'prediction': prediction,
        'confidence': float(confidence),
        'frames_analyzed': len(frames),
        'fake_probability': float(avg_prob),
        'real_probability': float(1 - avg_prob),
        'fake_frames_count': fake_frames,
        'fake_frames_percentage': float(fake_percentage)
    })

@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera capture"""
    global is_capturing
    
    with camera_lock:
        is_capturing = False
    
    print("📹 Camera capture stopped\n")
    return jsonify({'status': 'Camera stopped'})

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 DEEPFAKE DETECTION SERVER STARTING")
    print("="*70)
    print(f"📍 Model: {MODEL_PATH}")
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🌐 Server: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
