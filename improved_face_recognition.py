import cv2
import numpy as np
import os
import json
import base64
from datetime import datetime

class ImprovedFaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.recognition_threshold = 0.6  # Now using 0.6 for normalized vectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✅ Improved Face Recognition System with Unit Normalization Initialized")
    
    def load_known_faces(self, students):
        """Load face encodings from student database and normalize them"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        print(f"🔍 Loading face encodings for {len(students)} students...")
        
        loaded_count = 0
        for student in students:
            if student.face_encoding:
                try:
                    encoding = json.loads(student.face_encoding)
                    encoding = np.array(encoding)
                    # Normalize the encoding (in case it's not normalized)
                    norm = np.linalg.norm(encoding)
                    if norm > 0:
                        encoding = encoding / norm
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(student.name)
                    self.known_face_ids.append(student.student_id)
                    loaded_count += 1
                    print(f"✅ Loaded encoding for {student.name} ({student.student_id})")
                except Exception as e:
                    print(f"❌ Error loading encoding for {student.name}: {e}")
                    continue
            else:
                print(f"⚠️  No face encoding for {student.name}")
        
        print(f"✅ Successfully loaded {loaded_count}/{len(students)} face encodings")
        print(f"📊 Recognition threshold set to: {self.recognition_threshold}")

def enhanced_face_detection(self, image):
        # ... (same as before)
    def iou(self, box1, box2):
        # ... (same as before)

        def preprocess_face(self, face_image):
            """Enhanced face preprocessing with unit normalization"""
        try:
            # Convert to grayscale if needed
            if len(face_image.shape) == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            face_image = cv2.resize(face_image, (100, 100))
            
            # Apply multiple preprocessing steps
            # 1. Histogram equalization
            face_image = cv2.equalizeHist(face_image)
            
            # 2. Gaussian blur to reduce noise
            face_image = cv2.GaussianBlur(face_image, (3, 3), 0)
            
            # 3. Normalize pixel values to [0,1]
            face_image = face_image.astype(np.float32) / 255.0
            
            # 4. Flatten and normalize to unit length
            encoding = face_image.flatten()
            norm = np.linalg.norm(encoding)
            if norm > 0:
                encoding = encoding / norm
            
            return encoding
            
        except Exception as e:
            print(f"❌ Error preprocessing face: {e}")
            return None
def encode_face(self, image_path):
        # ... (same as before, but now preprocess_face returns normalized vector)

      def recognize_face(self, frame):
        """Recognize faces with normalized encodings"""
        try:
            # Enhanced face detection
            faces = self.enhanced_face_detection(frame)
            
            print(f"🔍 Enhanced detection found {len(faces)} face(s) in frame")
            
            face_names = []
            face_locations = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Preprocess current face (returns normalized encoding)
                current_encoding = self.preprocess_face(face_roi)
                
                if current_encoding is None:
                    face_names.append("Unknown")
                    face_locations.append((y, x+w, y+h, x))
                    continue
                
                name = "Unknown"
                best_distance = float('inf')
                best_match_index = -1
                
                # Compare with known faces (which are normalized)
                for i, known_encoding in enumerate(self.known_face_encodings):
                    try:
                        # Calculate Euclidean distance between normalized vectors
                        distance = np.linalg.norm(current_encoding - known_encoding)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match_index = i
                    except Exception as e:
                        print(f"❌ Error comparing encodings: {e}")
                        continue
                
                # Check if best match is below threshold
                if best_match_index != -1 and best_distance < self.recognition_threshold:
                    name = self.known_face_ids[best_match_index]
                    print(f"✅ Recognized: {name} (distance: {best_distance:.4f}, threshold: {self.recognition_threshold})")
                else:
                    if best_match_index != -1:
                        print(f"❌ No match (best distance: {best_distance:.4f}, threshold: {self.recognition_threshold})")
                    else:
                        print("❌ No known faces to compare with")
                
                face_names.append(name)
                face_locations.append((y, x+w, y+h, x))
            
            return face_names, face_locations
            
        except Exception as e:
            print(f"❌ Error in face recognition: {e}")
            import traceback
            traceback.print_exc()
            return [], []