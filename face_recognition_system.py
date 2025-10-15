import cv2
import numpy as np
import os
import json
import base64

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        # Load OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def load_known_faces(self, students):
        """Load face encodings from student database"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        print(f"🔍 Loading face encodings for {len(students)} students...")
        
        for student in students:
            if student.face_encoding:
                try:
                    encoding = json.loads(student.face_encoding)
                    self.known_face_encodings.append(np.array(encoding))
                    self.known_face_names.append(student.student_id)
                    print(f"✅ Loaded encoding for {student.name} ({student.student_id})")
                except Exception as e:
                    print(f"❌ Error loading encoding for {student.name}: {e}")
                    continue
            else:
                print(f"❌ No face encoding for {student.name}")
        
        print(f"✅ Successfully loaded {len(self.known_face_encodings)} face encodings")

    def encode_face(self, image_path):
        """Encode face from image file"""
        try:
            print(f"🔍 Encoding face from: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                print("❌ Image file does not exist")
                return None
                
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print("❌ Could not read image file - file may be corrupted or wrong format")
                return None
                
            print(f"✅ Image loaded successfully. Shape: {image.shape}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f"✅ Converted to grayscale. Shape: {gray.shape}")
            
            # Detect faces with different parameters
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            print(f"✅ Using cascade classifier: {face_cascade_path}")
            
            # Try multiple detection parameters
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            print(f"🔍 Face detection result: {len(faces)} face(s) found")
            
            if len(faces) > 0:
                # Use the largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = faces[0]
                print(f"✅ Using face at position: x={x}, y={y}, w={w}, h={h}")
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                print(f"✅ Face ROI shape: {face_roi.shape}")
                
                # Resize to standard size
                face_roi = cv2.resize(face_roi, (100, 100))
                print(f"✅ Resized face ROI shape: {face_roi.shape}")
                
                # Normalize pixel values
                face_roi = face_roi.astype(np.float32) / 255.0
                print(f"✅ Normalized face ROI. Min: {face_roi.min()}, Max: {face_roi.max()}")
                
                # Create encoding
                encoding = face_roi.flatten().tolist()
                print(f"✅ Face encoded successfully. Encoding length: {len(encoding)}")
                
                return json.dumps(encoding)
            else:
                print("❌ No faces detected in image. Trying alternative detection parameters...")
                
                # Try alternative parameters
                faces_alt = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
                
                if len(faces_alt) > 0:
                    print(f"✅ Alternative detection found {len(faces_alt)} face(s)")
                    # Process the first face found with alternative parameters
                    (x, y, w, h) = faces_alt[0]
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (100, 100))
                    face_roi = face_roi.astype(np.float32) / 255.0
                    encoding = face_roi.flatten().tolist()
                    print(f"✅ Face encoded with alternative parameters")
                    return json.dumps(encoding)
                else:
                    print("❌ No faces detected even with alternative parameters")
                    return None
                    
        except Exception as e:
            print(f"❌ Error encoding face: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def recognize_face(self, frame):
    """Recognize face from camera frame"""
    try:
        # Resize frame for faster processing (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert to grayscale
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        print(f"🔍 Detected {len(faces)} face(s) in frame")
        
        face_names = []
        face_locations = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            face_roi = face_roi.astype(np.float32) / 255.0
            current_encoding = face_roi.flatten()
            
            name = "Unknown"
            min_distance = float('inf')
            best_match_index = -1
            
            # Compare with known faces
            for i, known_encoding in enumerate(self.known_face_encodings):
                try:
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(current_encoding - known_encoding)
                    
                    # Lower distance = better match
                    if distance < min_distance:
                        min_distance = distance
                        best_match_index = i
                except Exception as e:
                    print(f"❌ Error comparing with known face {i}: {e}")
                    continue
            
            # Set threshold for recognition
            recognition_threshold = 0.6  # Adjust this threshold as needed
            if best_match_index != -1 and min_distance < recognition_threshold:
                name = self.known_face_names[best_match_index]
                print(f"✅ Recognized: {name} (distance: {min_distance:.4f})")
            else:
                if best_match_index != -1:
                    print(f"❌ Face not recognized (best distance: {min_distance:.4f}, threshold: {recognition_threshold})")
                else:
                    print(f"❌ No known faces to compare with")
            
            face_names.append(name)
            # Convert coordinates back to original scale and format
            face_locations.append((y*4, (x+w)*4, (y+h)*4, x*4))  # Multiply by 4 if using 0.25 scale
        
        return face_names, face_locations
        
    except Exception as e:
        print(f"❌ Error in face recognition: {e}")
        return [], []