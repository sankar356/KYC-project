import cv2
import face_recognition
import os
import sqlite3
import shutil
from datetime import datetime

def create_db_connection():
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    
    # Create tables for different document types
    c.execute('''CREATE TABLE IF NOT EXISTS aadhaar_documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  aadhaar_number TEXT,
                  name TEXT,
                  dob TEXT,
                  verification_date TIMESTAMP)''')
                  
    c.execute('''CREATE TABLE IF NOT EXISTS passport_documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  passport_number TEXT,
                  name TEXT,
                  date_of_birth TEXT,
                  verification_date TIMESTAMP)''')
                  
    c.execute('''CREATE TABLE IF NOT EXISTS licence_documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  dl_number TEXT,
                  name TEXT,
                  dob TEXT,
                  verification_date TIMESTAMP)''')
    
    return conn

def save_to_database(conn, document_data):
    cursor = conn.cursor()
    
    doc_type = document_data.get('document_type', '').lower()
    if 'aadhaar' in doc_type:
        cursor.execute("""
            INSERT INTO aadhaar_documents (aadhaar_number, name, dob, verification_date)
            VALUES (?, ?, ?, ?)
        """, (document_data.get('aadhaar_number'), document_data.get('name'),
              document_data.get('dob'), datetime.now()))
    
    elif 'passport' in doc_type:
        cursor.execute("""
            INSERT INTO passport_documents (passport_number, name, date_of_birth, verification_date)
            VALUES (?, ?, ?, ?)
        """, (document_data.get('passport_number'), document_data.get('name'),
              document_data.get('date_of_birth'), datetime.now()))
    
    elif 'licence' in doc_type or 'license' in doc_type:
        cursor.execute("""
            INSERT INTO licence_documents (dl_number, name, dob, verification_date)
            VALUES (?, ?, ?, ?)
        """, (document_data.get('dl_number'), document_data.get('name'),
              document_data.get('dob'), datetime.now()))
    
    conn.commit()

def cleanup_images(cropped_face_path):
    """Remove verified images from cropped_faces and saved_images"""
    try:
        # Remove cropped face
        if os.path.exists(cropped_face_path):
            os.remove(cropped_face_path)
        
        # Remove corresponding saved image
        saved_image_path = cropped_face_path.replace('cropped_faces', 'saved_images')
        if os.path.exists(saved_image_path):
            os.remove(saved_image_path)
    except Exception as e:
        print(f"Error cleaning up images: {e}")

def verify_face():
    # Initialize database connection
    conn = create_db_connection()
    
    # Load cropped faces
    cropped_face_encodings = []
    cropped_face_paths = []
    document_data = {}
    
    for filename in os.listdir("cropped_faces"):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join("cropped_faces", filename)
            image = face_recognition.load_image_file(image_path)
            
            encoding = face_recognition.face_encodings(image)
            if encoding:
                cropped_face_encodings.append(encoding[0])
                cropped_face_paths.append(image_path)
                
                # Try to load corresponding document data
                doc_path = os.path.join("temp_documents", f"{os.path.splitext(filename)[0]}.json")
                if os.path.exists(doc_path):
                    import json
                    with open(doc_path, 'r') as f:
                        document_data[image_path] = json.load(f)

    if not cropped_face_encodings:
        print("No cropped faces found for verification")
        return
    
    print(f"✅ Loaded {len(cropped_face_encodings)} cropped faces")
    
    # Open webcam
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Process frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Match against cropped faces
            matches = face_recognition.compare_faces(cropped_face_encodings, face_encoding)
            if True in matches:
                # Get the matched face index
                match_index = matches.index(True)
                matched_face_path = cropped_face_paths[match_index]
                
                # Get corresponding document data
                doc_data = document_data.get(matched_face_path)
                if doc_data:
                    # Save to database
                    save_to_database(conn, doc_data)
                    print(f"✅ Document data saved to database")
                    
                    # Cleanup images
                    cleanup_images(matched_face_path)
                    
                    # Remove from tracking lists
                    cropped_face_encodings.pop(match_index)
                    cropped_face_paths.pop(match_index)
                    
                    if not cropped_face_encodings:
                        print("All faces verified!")
                        video_capture.release()
                        cv2.destroyAllWindows()
                        conn.close()
                        return
            
            # Draw face box
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow('Face Verification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    verify_face()
