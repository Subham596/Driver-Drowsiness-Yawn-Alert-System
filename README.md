# Driver Drowsiness & Yawn Alert System

A real-time computer vision driver safety system built with **Python**, **OpenCV**, **dlib**, **SciPy**, and **pygame**. Designed for edge deployment (NVIDIA Jetson) and standard webcams, it tracks 68 facial landmarks frame-by-frame to detect prolonged eye closure and yawning, triggering immediate audio alerts.

## Key Features
- **68-Point Facial Landmark Tracking:** Uses `dlib`'s facial landmark predictor to isolate eye and mouth coordinates in real time.
- **Eye Aspect Ratio (EAR):** Calculates vertical-to-horizontal eye landmark distances via `SciPy` to detect sustained eye closure and micro-sleep.
- **Mouth Aspect Ratio (MAR):** Monitors mouth landmark geometry across consecutive frames to detect yawning events.
- **Automatic Model Setup & Audio Alerts:** Automatically fetches the 68-point predictor on first run and plays low-latency warning alerts via `pygame.mixer`.
- **Live FPS Telemetry:** Displays real-time frame rate, EAR/MAR values, and alert state overlays on the video feed.

## Project Structure
```text
Driver-Drowsiness-Yawn-Alert-System/
├── main.py             # Real-time webcam detection loop, EAR/MAR logic & HUD
├── setup_model.py      # Automatic 68-point dlib model downloader & extractor
├── alarm.wav           # Two-tone audio alert waveform
├── requirements.txt    # Python dependencies
├── run.bat             # One-click Windows installer & launcher
├── .gitignore          # Git ignore rules for large binary model files
└── README.md           # Project documentation
```

## Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the system** *(automatically downloads `shape_predictor_68_face_landmarks.dat` on first launch)*:
   ```bash
   python main.py
   ```
   *(On Windows, you can also double-click `run.bat`.)*
3. Press **`q`** inside the video window to quit cleanly.

## Push to GitHub
```bash
git init
git add .
git commit -m "Add Driver Drowsiness and Yawn Alert System"
git branch -M main
git remote add origin https://github.com/Subham586/Driver-Drowsiness-Yawn-Alert-System.git
git push -u origin main
```
