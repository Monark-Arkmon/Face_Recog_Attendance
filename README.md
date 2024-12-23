#👨 Face Recognition Attendance System

A web-based attendance system using facial recognition, built with React (frontend) and FastAPI (backend).

## Features
- Face registration with multiple sample captures
- Real-time face recognition
- Automated attendance recording
- Daily attendance tracking
- User-friendly interface

## Setup Instructions

### Backend Setup
1. Clone backend repository:
```bash
git clone https://github.com/Monark-Arkmon/Face_Recog_Attendance_Backend.git
cd Face_Recog_Attendance_Backend
```

2. Install dependencies:
```bash
pip install fastapi uvicorn opencv-python numpy scikit-learn pandas python-multipart
```

3. Start backend server:
```bash
python -m uvicorn app:app --reload
```
Server runs at http://localhost:8000

### Frontend Setup
1. Clone frontend repository:
```bash
git clone https://github.com/Monark-Arkmon/Face_Recog_Attendance_Frontend.git
cd Face_Recog_Attendance_Frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start frontend application:
```bash
npm start
```
Application runs at http://localhost:3000

## Usage Guidelines

### Face Registration
1. Select "Register New Face"
2. Enter person's name
3. Ensure:
   - Well-lit face
   - Dark background preferred
   - Face directly facing camera
   - Stay still during capture
4. System captures 5 images automatically

### Taking Attendance
1. Select "Recognize Face"
2. Position face in frame
3. Click "Take Attendance"
4. System records attendance with timestamp

### Important Notes
- Face recognition works best in well-lit areas with dark backgrounds
- Maintain consistent lighting conditions
- Avoid excessive movement during registration
- Keep 50-100cm distance from camera
- Ensure clear facial visibility without obstructions
