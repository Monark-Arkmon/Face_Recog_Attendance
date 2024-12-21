from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import os
import csv
from typing import List
import uvicorn
import pandas as pd
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnhancedFaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        self.face_size = (75, 100)
        self.faces = np.array([]).reshape(0, self.face_size[0] * self.face_size[1])
        self.labels = []
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self._initialize_model()
        self.confidence_threshold = 0.6

    def _initialize_model(self):
        try:
            # Initialize or load existing model
            if os.path.exists('data/faces_data.pkl') and os.path.exists('data/names.pkl'):
                with open('data/names.pkl', 'rb') as w:
                    self.labels = pickle.load(w)
                with open('data/faces_data.pkl', 'rb') as f:
                    self.faces = pickle.load(f)
                
                if len(self.faces) > 0 and len(self.labels) > 0:
                    expected_size = self.face_size[0] * self.face_size[1]
                    self.faces = np.array(self.faces)
                    if self.faces.shape[1] != expected_size:
                        logging.error(f"Stored face data has wrong shape. Expected {expected_size}, got {self.faces.shape[1]}")
                        self.faces = np.array([]).reshape(0, expected_size)
                        self.labels = []
                    elif len(self.faces) != len(self.labels):
                        logging.error(f"Inconsistent number of samples: faces ({len(self.faces)}) and labels ({len(self.labels)})")
                        self.faces = np.array([]).reshape(0, expected_size)
                        self.labels = []
                    else:
                        self.knn.fit(self.faces, self.labels)
                        logging.info("Model loaded and trained with existing data")
                else:
                    self.faces = np.array([]).reshape(0, self.face_size[0] * self.face_size[1])
                    self.labels = []
                    logging.info("Initialized new model")
            else:
                self.faces = np.array([]).reshape(0, self.face_size[0] * self.face_size[1])
                self.labels = []
                logging.info("Initialized new model")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            self.faces = np.array([]).reshape(0, self.face_size[0] * self.face_size[1])
            self.labels = []

    def preprocess_face(self, face):
        try:
            if face is None or face.size == 0:
                logging.warning("Face is None or empty")
                return None
                
            face_resized = cv2.resize(face, self.face_size)
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_array = np.array(face_gray).flatten()
            
            expected_size = self.face_size[0] * self.face_size[1]
            if face_array.shape[0] != expected_size:
                logging.error(f"Preprocessed face has wrong shape. Expected {expected_size}, got {face_array.shape[0]}")
                return None
                
            return face_array
            
        except Exception as e:
            logging.error(f"Error in preprocess_face: {e}")
            return None

    def detect_faces(self, frame):
        try:
            if frame is None:
                logging.warning("Frame is None")
                return [], None
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )
            
            logging.info(f"Detected {len(faces)} faces")
            return faces, gray
            
        except Exception as e:
            logging.error(f"Error in detect_faces: {e}")
            return [], None

    def get_prediction_confidence(self, face_encoding):
        distances, indices = self.knn.kneighbors([face_encoding])
        mean_distance = np.mean(distances[0])
        confidence = 1 / (1 + mean_distance)
        return confidence

    def process_frame(self, frame):
        if len(self.faces) == 0 or len(self.labels) == 0:
            logging.warning("No training data available")
            return []
            
        faces, gray = self.detect_faces(frame)
        results = []
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            processed_face = self.preprocess_face(face)
            if processed_face is not None:
                prediction = self.knn.predict([processed_face])
                confidence = self.get_prediction_confidence(processed_face)
                results.append({"name": prediction[0], "confidence": confidence})
        
        return results

face_system = EnhancedFaceRecognitionSystem()

@app.post("/register-face/")
async def register_face(name: str = Form(...), images: List[UploadFile] = File(...)):
    logging.info(f"Registering face for {name} with {len(images)} images")
    faces_data = []
    
    for image in images:
        contents = await image.read()
        nparr = np.fromstring(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        faces, gray = face_system.detect_faces(frame)
        logging.info(f"Detected {len(faces)} faces in the image")
        
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            processed_face = face_system.preprocess_face(face)
            if processed_face is not None:
                faces_data.append(processed_face)
            else:
                logging.warning("Processed face is None")
        else:
            logging.warning(f"Expected 1 face, but detected {len(faces)} faces")
    
    logging.info(f"Collected {len(faces_data)} valid face samples for registration")
    
    if len(faces_data) >= 5:
        faces_data = np.array(faces_data)
        try:
            if os.path.exists('data/names.pkl'):
                with open('data/names.pkl', 'rb') as f:
                    names = pickle.load(f)
                if not isinstance(names, list):
                    names = []
                names.extend([name] * len(faces_data))
            else:
                names = [name] * len(faces_data)

            with open('data/names.pkl', 'wb') as f:
                pickle.dump(names, f)

            if os.path.exists('data/faces_data.pkl'):
                with open('data/faces_data.pkl', 'rb') as f:
                    existing_faces = pickle.load(f)

                existing_faces = np.array(existing_faces)
                if existing_faces.shape[1] != faces_data.shape[1]:
                    logging.error(f"Existing face data has wrong shape. Expected {faces_data.shape[1]}, got {existing_faces.shape[1]}")
                    existing_faces = np.array([]).reshape(0, faces_data.shape[1])

                faces_data = np.vstack((existing_faces, faces_data))
            else:
                faces_data = faces_data

            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces_data, f)

            face_system._initialize_model()

            logging.info(f"Successfully registered {len(faces_data)} faces for {name}")
            return {"message": f"Successfully registered {len(faces_data)} faces for {name}"}
        except Exception as e:
            logging.error(f"Error during registration: {e}")
            return {"message": "An error occurred during registration"}

    logging.error("Insufficient face data for registration")
    return {"message": "Insufficient face data for registration. Please provide more images."}

@app.post("/recognize-face/")
async def recognize_face(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.fromstring(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = face_system.process_frame(frame)
        
        if results:
            timestamp = datetime.now()
            date = timestamp.strftime("%Y-%m-%d")
            time = timestamp.strftime("%H:%M:%S")
                
            attendance_file = f"Attendance/Attendance_{date}.csv"
            
            high_confidence_results = [r for r in results]
            
            for result in high_confidence_results:
                attendance = [result["name"], time]
                
                if not os.path.exists(attendance_file):
                    with open(attendance_file, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["NAME", "TIME"])
                        writer.writerow(attendance)
                else:
                    with open(attendance_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(attendance)
            
            logging.info(f"Attendance recorded for {', '.join([r['name'] for r in high_confidence_results])} at {time}")
            
            return {"results": results}
        else:
            logging.warning("No faces recognized")
            return {"results": []}
            
    except Exception as e:
        logging.error(f"An error occurred while recognizing face: {e}")
        return {"message": "An error occurred while recognizing face"}

@app.get("/attendance/{date}")
async def get_attendance(date: str):
    try:
        attendance_file = f"Attendance/Attendance_{date}.csv"
        if not os.path.exists(attendance_file):
            return {"attendance": []}
            
        df = pd.read_csv(attendance_file)
        

        attendance_records = []
        for _, row in df.iterrows():
            attendance_records.append({
                "name": row["NAME"],
                "time": row["TIME"]
            })
            
        return {"attendance": attendance_records}
    except Exception as e:
        logging.error(f"An error occurred while fetching attendance data for {date}: {e}")
        return {"attendance": []}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)