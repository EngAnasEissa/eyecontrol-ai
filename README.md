# EyeControl AI 👁️🤖

**An Intelligent Vision-Based Human-Computer Interaction System Using Eye Tracking and Voice Confirmation**  
*نظام ذكي للتفاعل بين الإنسان والحاسوب معتمد على الرؤية الحاسوبية وتتبع حركة العين والتأكيد الصوتي*

---

## 1. Project Overview / نبذة عن المشروع
**EyeControl AI** is an academic Computer Vision project designed to allow users (especially individuals with motor impairments or hands-busy scenarios) to interact with a computer using eye gaze, dwell detection, and voice confirmation.

The project is built in **strict modular phases** to maintain clarity, academic rigor, and clean engineering.

---
cd "c:\Users\HP\Desktop\eyecontrol-ai"
$env:PYTHONPATH = "."
C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe -m app.main



## 2. Phase 1 Accomplishments / ما تم إنجازه في المرحلة الأولى
In **Phase 1**, we established the foundational computer vision pipeline:
1. Captured real-time video frames from the webcam using **OpenCV**.
2. Integrated **MediaPipe Face Landmarker** to extract 478 3D facial landmarks.
3. Implemented safe resource management and graceful shutdown (`Q` or `ESC`).
4. Displayed basic status HUD (FPS, face count, camera status).

---

## 3. Phase 2: Eye & Iris Feature Extraction / المرحلة الثانية: استخراج خصائص العين والبؤبؤ
In **Phase 2**, we built upon the facial landmarks to extract geometric eye features without gaze estimation:
1. **Eye Region Localization**: Isolated anatomical left and right eye contours.
2. **Iris Landmark Extraction**: Extracted iris landmarks (5 points per iris: center + 4 perimeter points).
3. **Iris Center Calculation**: Computed geometric center by averaging iris points.
4. **Normalized Iris Position**: Calculated relative iris coordinates within eye bounding boundaries $[0.0, 1.0]$.
5. **Eye Aspect Ratio (EAR)**: Computed vertical-to-horizontal ratio measuring eye openness/blinking.
6. **Eye Dimensions**: Calculated normalized eye width and height.
7. **Basic Head Reference**: Calculated inter-eye distance and face center.

### Conceptual Pipeline / مخطط تدفق البيانات
```text
Webcam (الكاميرا)
   ↓
MediaPipe Face Landmarks (نقاط معالم الوجه 478 نقطة)
   ↓
Eye + Iris Landmarks (نقاط العين والقرنية)
   ↓
Geometric Feature Extraction (استخراج الخصائص الهندسية)
   ↓
Normalized Eye Features (خصائص العين المعيارية)
   ↓
Future Calibration (المعايرة في المراحل القادمة)
```

---

## 4. Scientific Principle & Landmark Math / المبدأ العلمي والمعادلات الرياضية

> [!IMPORTANT]
> **MediaPipe does NOT predict the final screen position of the user's gaze.**  
> MediaPipe is a pre-trained computer vision model that provides 3D facial landmarks. **EyeControl AI** extracts geometric features from these landmarks. These normalized features will later be used during a personalized calibration phase to learn the mapping to screen coordinates.

### Key Mathematical Formulas / المعادلات الرياضية:

1. **Iris Center / مركز بؤبؤ العين**:
   $$\text{Center}_x = \frac{1}{N} \sum_{i=1}^{N} x_i, \quad \text{Center}_y = \frac{1}{N} \sum_{i=1}^{N} y_i$$
   *(Calculated from MediaPipe iris landmarks: left `[473..477]`, right `[468..472]`)*

2. **Normalized Iris Position / الموضع المعياري للبؤبؤ**:
   $$\text{norm}_x = \frac{\text{iris}_x - \min(x_{\text{eye}})}{\max(x_{\text{eye}}) - \min(x_{\text{eye}})}, \quad \text{norm}_y = \frac{\text{iris}_y - \min(y_{\text{eye}})}{\max(y_{\text{eye}}) - \min(y_{\text{eye}})}$$
   - `0.0` → Iris positioned toward one corner.
   - `0.5` → Iris centered.
   - `1.0` → Iris positioned toward the opposite corner.
   *(Note: These are normalized eye-relative ratios, NOT screen pixel coordinates).*

3. **Eye Aspect Ratio (EAR) / نسبة أبعاد العين**:
   Based on the formula by Soukupová & Čech:
   $$\text{EAR} = \frac{\|P_{\text{top1}} - P_{\text{bottom1}}\| + \|P_{\text{top2}} - P_{\text{bottom2}}\|}{2 \cdot \|P_{\text{outer}} - P_{\text{inner}}\|}$$
   - When the eye is open: $\text{EAR} \approx 0.25 - 0.35$.
   - When the eye blinks/closes: $\text{EAR} < 0.18$.

---

## 5. Technology Stack / التقنيات المستخدمة
- **Python 3.11+** (Tested on Python 3.12)
- **OpenCV (`opencv-python`)**: Video capture, frame transformations, visual HUD overlay.
- **MediaPipe (`mediapipe`)**: Pre-trained Tasks API for 478 face and iris landmarks.
- **NumPy**: Matrix and coordinate operations.

---

## 6. Project Structure / هيكلية المشروع
```text
eyecontrol-ai/
│
├── app/
│   ├── __init__.py
│   ├── main.py                     # Entry point (Camera loop + HUD rendering)
│   └── vision/
│       ├── __init__.py
│       ├── camera.py               # Reusable OpenCV Camera manager
│       ├── face_landmarks.py       # MediaPipe Tasks FaceLandmarker wrapper
│       └── eye_features.py         # Phase 2: Iris center, Normalized position, EAR
│
├── models/
│   └── face_landmarker.task        # Official Google MediaPipe pre-trained model bundle
│
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

---

## 7. Installation & Setup / خطوات التثبيت والتشغيل

### Step 1: Create Virtual Environment / إنشاء بيئة افتراضية
Open PowerShell or Command Prompt in the project folder:
```powershell
python -m venv .venv
```
Activate the environment:
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Step 2: Install Dependencies / تثبيت المكتبات
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Program / تشغيل البرنامج
```powershell
python app/main.py
```

### Step 4: Exit the Program / إغلاق البرنامج
Press the **`Q`** key or **`ESC`** while focused on the camera window to safely shut down.

---

## 8. Common Troubleshooting & Permissions / استكشاف الأخطاء الشائعة

1. **Camera Permission Error (مشكلة إذن الكاميرا):**
   - On Windows: Go to **Settings > Privacy & Security > Camera**.
   - Make sure **"Camera access"** and **"Let desktop apps access your camera"** are turned **ON**.

2. **Camera Busy or Not Found (الكاميرا مشغولة):**
   - Ensure no other application (Zoom, Teams, Skype, or browser) is currently using the camera.
   - If using an external webcam, try changing `device_index=0` to `device_index=1` in `app/main.py`.

3. **Missing Model File (`models/face_landmarker.task`):**
   - The official model bundle is automatically loaded from `models/face_landmarker.task`.
   - If missing, download it from Google's official storage:
     `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
     and place it inside the `models/` directory.

---

## 9. Future Roadmap / المراحل القادمة
- **Phase 3:** Calibration & Gaze Mapping (Learning user-specific screen coordinates).
- **Phase 4:** Dwell-Time Detection & Eye-Clicking.
- **Phase 5:** Speech Recognition & Voice Confirmation.
- **Phase 6:** HCI Integration & Desktop Automation.
