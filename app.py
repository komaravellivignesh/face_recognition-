from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
from datetime import datetime, timedelta
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename

from improved_face_recognition import ImprovedFaceRecognitionSystem
face_system = ImprovedFaceRecognitionSystem()

from database import db, College, Admin, Student, Attendance
from face_recognition_system import FaceRecognitionSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cogniface-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cogniface.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads/student_photos'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize face recognition system
face_system = FaceRecognitionSystem()

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

def init_db():
    """Initialize database with sample data"""
    with app.app_context():
        db.create_all()
        
        # Create sample colleges if they don't exist
        if College.query.count() == 0:
            college1 = College(name="ABC Engineering College", code="ABCEC", address="123 College Road, City")
            college2 = College(name="XYZ University", code="XYZU", address="456 University Avenue, Town")
            db.session.add(college1)
            db.session.add(college2)
            db.session.commit()
            
            # Create sample admin accounts
            admin1 = Admin(username='admin_abc', college_id=college1.id, phone='+1234567890')
            admin1.set_password('admin123')
            
            admin2 = Admin(username='admin_xyz', college_id=college2.id, phone='+0987654321')
            admin2.set_password('admin123')
            
            db.session.add(admin1)
            db.session.add(admin2)
            db.session.commit()
            
            print("Sample data created:")
            print("College ABC: admin_abc / admin123")
            print("College XYZ: admin_xyz / admin123")

@app.route('/')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')
@app.route('/test-recognition/<student_id>')
@login_required
def test_recognition(student_id):
    """Test recognition with current student"""
    student = Student.query.filter_by(
        student_id=student_id,
        college_id=current_user.college_id
    ).first_or_404()
    
    # Test with the student's own photo
    if student.photo_path and os.path.exists(student.photo_path):
        # Load test image
        test_image = cv2.imread(student.photo_path)
        
        if test_image is not None:
            # Test recognition on the same photo (should match perfectly)
            face_names, face_locations = face_system.recognize_face(test_image)
            
            result = {
                'student': student,
                'test_image_loaded': True,
                'faces_detected': len(face_locations),
                'recognition_result': face_names,
                'distance_info': 'Test completed'
            }
        else:
            result = {
                'student': student,
                'test_image_loaded': False,
                'error': 'Could not load test image'
            }
    else:
        result = {
            'student': student,
            'test_image_loaded': False,
            'error': 'Photo path does not exist'
        }
    
    return render_template('test_recognition.html', result=result, college=current_user.college)

@app.route('/reencode-all-faces')
@login_required
def reencode_all_faces():
    """Re-encode all faces with the new system"""
    students = Student.query.filter_by(college_id=current_user.college_id).all()
    
    success_count = 0
    failed_count = 0
    
    for student in students:
        if student.photo_path and os.path.exists(student.photo_path):
            new_encoding = face_system.encode_face(student.photo_path)
            if new_encoding:
                student.face_encoding = new_encoding
                success_count += 1
                print(f"✅ Re-encoded: {student.name}")
            else:
                failed_count += 1
                print(f"❌ Failed: {student.name}")
    
    db.session.commit()
    
    # Reload known faces
    students = Student.query.filter_by(college_id=current_user.college_id).all()
    face_system.load_known_faces(students)
    
    flash(f'Re-encoded {success_count} faces successfully. {failed_count} failed.', 'success')
    return redirect(url_for('debug_students'))
@app.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    
    admin = Admin.query.filter_by(username=username).first()
    
    if admin and admin.check_password(password):
        login_user(admin)
        
        # Load face encodings for this college after login
        students = Student.query.filter_by(college_id=admin.college_id).all()
        face_system.load_known_faces(students)
        
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid credentials', 'error')
        return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    college = current_user.college
    students = Student.query.filter_by(college_id=college.id).all()
    today_attendance = Attendance.query.join(Student).filter(
        Student.college_id == college.id,
        Attendance.date == datetime.utcnow().date()
    ).all()
    
    return render_template('dashboard.html',
                         college=college,
                         total_students=len(students),
                         present_today=len(today_attendance),
                         students=students[:5])

@app.route('/add-student', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        email = request.form.get('email')
        photo = request.files.get('photo')
        
        if photo and student_id and name:
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            filename = secure_filename(f"{student_id}_{photo.filename}")
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the photo
            photo.save(photo_path)
            print(f"✅ Photo saved to: {photo_path}")
            print(f"✅ Photo exists: {os.path.exists(photo_path)}")
            
            # Encode face
            face_encoding = face_system.encode_face(photo_path)
            
            if face_encoding is None:
                flash(f'Warning: Could not detect face in the uploaded photo for {name}. Student was added but face recognition may not work. Please try with a clearer front-facing photo.', 'warning')
            
            student = Student(
                college_id=current_user.college_id,
                student_id=student_id,
                name=name,
                email=email,
                photo_path=photo_path,
                face_encoding=face_encoding
            )
            
            db.session.add(student)
            db.session.commit()
            
            # Reload known faces for current college
            students = Student.query.filter_by(college_id=current_user.college_id).all()
            face_system.load_known_faces(students)
            
            if face_encoding:
                flash(f'Student {name} added successfully with face encoding!', 'success')
            else:
                flash(f'Student {name} added but face encoding failed. Please use the debug page to re-encode.', 'warning')
            
            return redirect(url_for('dashboard'))
        else:
            flash('Please fill all fields and upload a photo', 'error')
    
    return render_template('add_student.html', college=current_user.college)

@app.route('/mark-attendance', methods=['GET', 'POST'])
@login_required
def mark_attendance():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        
        student = Student.query.filter_by(
            student_id=student_id,
            college_id=current_user.college_id
        ).first()
        
        if student:
            today = datetime.utcnow().date()
            attendance = Attendance.query.filter_by(
                student_id=student.id,
                date=today
            ).first()
            
            if not attendance:
                # Check-in
                attendance = Attendance(student_id=student.id, check_in=datetime.utcnow())
                db.session.add(attendance)
                action = 'Check-in'
            else:
                # Check-out
                attendance.check_out = datetime.utcnow()
                attendance.calculate_duration()
                action = 'Check-out'
            
            db.session.commit()
            return jsonify({'success': True, 'student': student.name, 'action': action})
        else:
            return jsonify({'success': False, 'error': 'Student not found'})
    
    return render_template('attendance.html', college=current_user.college)

@app.route('/live-attendance')
@login_required
def live_attendance():
    return render_template('live_attendance.html', college=current_user.college)

@app.route('/recognize-face', methods=['POST'])
@login_required
def recognize_face():
    try:
        image_data = request.json.get('image')
        auto_capture = request.json.get('auto_capture', False)
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data'})
        
        # Convert base64 to image
        format, imgstr = image_data.split(';base64,')
        nparr = np.frombuffer(base64.b64decode(imgstr), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Could not decode image'})
        
        # Recognize face
        face_names, face_locations = face_system.recognize_face(frame)
        
        if face_names and face_names[0] != "Unknown":
            student_id = face_names[0]
            student = Student.query.filter_by(
                student_id=student_id,
                college_id=current_user.college_id
            ).first()
            
            if student:
                today = datetime.utcnow().date()
                attendance = Attendance.query.filter_by(
                    student_id=student.id,
                    date=today
                ).first()
                
                if not attendance:
                    # Check-in
                    attendance = Attendance(student_id=student.id, check_in=datetime.utcnow())
                    db.session.add(attendance)
                    action = 'Check-in'
                else:
                    # Check-out
                    attendance.check_out = datetime.utcnow()
                    attendance.calculate_duration()
                    action = 'Check-out'
                
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'student_name': student.name,
                    'student_id': student.student_id,
                    'action': action,
                    'face_location': face_locations[0] if face_locations else None,
                    'auto_capture': auto_capture
                })
        
        return jsonify({'success': False, 'error': 'Face not recognized'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
@app.route('/attendance-report')
@login_required
def attendance_report():
    college = current_user.college
    students = Student.query.filter_by(college_id=college.id).all()
    
    attendance_data = []
    for student in students:
        attendances = Attendance.query.filter_by(student_id=student.id).all()
        total_days = len(attendances)
        present_days = len([a for a in attendances if a.status == 'PRESENT'])
        
        percentage = round((present_days / total_days * 100) if total_days > 0 else 0, 2)
        
        attendance_data.append({
            'student': student,
            'total_days': total_days,
            'present_days': present_days,
            'attendance_percentage': percentage
        })
    
    return render_template('attendance_report.html',
                         attendance_data=attendance_data,
                         college=college)

@app.route('/debug-students')
@login_required
def debug_students():
    """Debug page to check student photos and encodings"""
    students = Student.query.filter_by(college_id=current_user.college_id).all()
    
    debug_info = []
    for student in students:
        photo_exists = os.path.exists(student.photo_path) if student.photo_path else False
        has_encoding = bool(student.face_encoding)
        
        debug_info.append({
            'student': student,
            'photo_exists': photo_exists,
            'photo_path': student.photo_path,
            'has_encoding': has_encoding
        })
    
    return render_template('debug_students.html', debug_info=debug_info, college=current_user.college)
@app.route('/debug-face/<student_id>')
@login_required
def debug_face(student_id):
    """Debug face encoding for a specific student"""
    student = Student.query.filter_by(
        student_id=student_id,
        college_id=current_user.college_id
    ).first_or_404()
    
    debug_info = {
        'student': student,
        'photo_exists': os.path.exists(student.photo_path) if student.photo_path else False,
        'has_encoding': bool(student.face_encoding),
        'photo_path': student.photo_path
    }
    
    if debug_info['photo_exists']:
        # Test encoding with the improved system
        test_encoding = face_system.encode_face(student.photo_path)
        debug_info['test_encoding_success'] = bool(test_encoding)
        debug_info['test_encoding'] = test_encoding
    
    return render_template('debug_face.html', debug_info=debug_info, college=current_user.college)
@app.route('/reencode-face/<int:student_id>', methods=['POST'])
@login_required
def reencode_face(student_id):
    """Re-encode face for a student"""
    student = Student.query.filter_by(
        id=student_id,
        college_id=current_user.college_id
    ).first_or_404()
    
    if student.photo_path and os.path.exists(student.photo_path):
        # Re-encode face
        face_encoding = face_system.encode_face(student.photo_path)
        
        if face_encoding:
            student.face_encoding = face_encoding
            db.session.commit()
            
            # Reload known faces
            students = Student.query.filter_by(college_id=current_user.college_id).all()
            face_system.load_known_faces(students)
            
            flash(f'Face re-encoded successfully for {student.name}!', 'success')
        else:
            flash(f'Failed to re-encode face for {student.name}. Please check the photo.', 'error')
    else:
        flash(f'Photo not found for {student.name}.', 'error')
    
    return redirect(url_for('debug_students'))

@app.route('/test-camera')
@login_required
def test_camera():
    """Test page to verify camera functionality"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Test</title>
    </head>
    <body>
        <h1>Camera Test Page</h1>
        <video id="video" width="640" height="480" autoplay muted playsinline></video>
        <br>
        <button onclick="startCamera()">Start Camera</button>
        <button onclick="stopCamera()">Stop Camera</button>
        
        <script>
            let stream = null;
            
            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 640, height: 480 },
                        audio: false 
                    });
                    document.getElementById('video').srcObject = stream;
                    alert('Camera started successfully!');
                } catch (err) {
                    alert('Camera error: ' + err.message);
                }
            }
            
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                    document.getElementById('video').srcObject = null;
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.before_request
def load_face_data():
    """Load face encodings for the current user's college"""
    if current_user.is_authenticated:
        students = Student.query.filter_by(college_id=current_user.college_id).all()
        face_system.load_known_faces(students)
if __name__ == '__main__':
    # Initialize database
    init_db()
    
    print("CogniFace Flask Application Starting...")
    print("Access the application at: http://127.0.0.1:5000")
    print("Sample logins:")
    print("  College ABC: admin_abc / admin123")
    print("  College XYZ: admin_xyz / admin123")
    print("\nDebug Pages:")
    print("  Camera Test: http://127.0.0.1:5000/test-camera")
    print("  Student Debug: http://127.0.0.1:5000/debug-students")
    
    app.run(debug=True, host='127.0.0.1', port=5000)

# Start Flask shell
# flask shell

from app import db, Student, face_system
from improved_face_recognition import WorkingFaceRecognitionSystem

# Get your student
student = Student.query.filter_by(student_id='23241-cs-046').first()

if student and student.photo_path:
    print(f"Re-encoding face for: {student.name}")
    
    # Use the new system
    new_system = WorkingFaceRecognitionSystem()
    new_encoding = new_system.encode_face(student.photo_path)
    
    if new_encoding:
        student.face_encoding = new_encoding
        db.session.commit()
        print("✅ Face re-encoded successfully!")
        
        # Reload known faces
        students = Student.query.all()
        new_system.load_known_faces(students)
        print("✅ Known faces reloaded!")
    else:
        print("❌ Face encoding failed!")
else:
    print("❌ Student or photo not found!")