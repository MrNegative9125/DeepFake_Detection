🕵️‍♂️ DeepFake Detection System (Flask + PyTorch)

A deep learning–based DeepFake Detection web application built using PyTorch and Flask, designed to identify manipulated (fake) images/videos by detecting subtle forensic artifacts left during deepfake generation.

This project focuses on generalization-aware deepfake detection, not identity recognition.

🚀 Features

🔍 DeepFake detection using a custom CNN model

🧠 PyTorch-based inference pipeline

🌐 Flask web interface

📤 Image / video upload support

📊 Real vs Fake prediction with confidence

⚙️ Modular & extensible architecture

💻 CPU inference (deployment-friendly)

🧠 Model Overview

The model is trained to detect forensic artifacts, not facial identity.

Detects artifacts from:

GAN-generated deepfakes

Face swapping

Face reenactment

Synthetic video/image manipulation

Why this matters:

✔️ Reduces identity overfitting
✔️ Improves real-world generalization
✔️ More robust against unseen faces

🏗️ Project Structure
DeepFake_Detection/
│
├── app.py                  # Flask application entry point
├── model/
│   └── model.pth           # Trained PyTorch model
│
├── templates/
│   └── index.html          # Frontend UI
│
├── static/
│   └── uploads/            # Uploaded files
│
├── utils/
│   ├── preprocess.py       # Image preprocessing
│   └── inference.py        # Model inference logic
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

🛠️ Tech Stack

Backend: Flask

Deep Learning: PyTorch

Image Processing: OpenCV, PIL

Frontend: HTML, CSS

Deployment: Hugging Face / Render (CPU)

⚙️ Installation (Local Setup)
1️⃣ Clone the repository
git clone https://github.com/MrNegative9125/DeepFake_Detection.git
cd DeepFake_Detection

2️⃣ Create virtual environment
conda create -n deepfake python=3.9
conda activate deepfake

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Flask app
python app.py

5️⃣ Open in browser
http://127.0.0.1:5000

🌐 Deployment (Free Options)
✅ Recommended: Hugging Face Spaces

Free CPU hosting

ML-friendly

Ideal for demos & portfolios

Flask can be adapted to Gradio for easier deployment.

⚠️ Alternative: Render (Free Tier)

Flask supported

Limited RAM & cold starts

Best for lightweight inference

📌 Limitations

❌ Free tier does not support GPU

⏳ Video inference may be slow on CPU

📦 File upload size is limited

📊 Output Example
Input	Prediction	Confidence
Real Image	REAL	92%
DeepFake Image	FAKE	87%
🔒 Ethical Disclaimer

This project is intended strictly for educational, research, and forensic analysis purposes.

❌ Do NOT use for:

Surveillance

Harassment

Misrepresentation

Malicious profiling

✔️ Use responsibly.

👨‍💻 Author

MrNegative
GitHub: MrNegative9125

⭐ Acknowledgements

PyTorch

OpenCV

Research papers on DeepFake forensics

Open-source ML community

📜 License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute with attribution.
