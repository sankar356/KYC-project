import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import os
import sqlite3
import json
from datetime import datetime
import time

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks for EAR calculation (based on MediaPipe indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks for mouth opening detection
# MOUTH_LANDMARKS = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]

class AdvancedLivenessDetector:
    def __init__(self):
        # Blink detection
        self.ear_threshold = 0.25
        self.blink_consec_frames = 3
        self.blink_counter = 0
        self.blinks_detected = 0
        self.required_blinks = 3
        
        # # Mouth movement detection
        # self.mouth_threshold = 15.0
        # self.mouth_history = []
        # self.mouth_movements = 0
        # self.required_mouth_movements = 2
        
        # Head pose tracking (more strict)
        self.head_pose_history = []
        self.head_movements = 0
        self.required_head_movements = 3
        self.head_movement_threshold = 15.0
        
        # Reflection detection (detect screen/photo reflections)
        self.reflection_history = []
        self.reflection_threshold = 0.8
        self.is_photo_detected = False
        
        # Depth analysis (photos are flat)
        self.depth_history = []
        self.depth_variance_threshold = 50.0  # Reduced threshold for better accuracy
        self.depth_analysis_frames = 0
        
        # Frame difference analysis
        self.prev_frame = None
        self.frame_diff_history = []
        self.min_frame_difference = 500.0
        
        # Time-based checks
        self.start_time = time.time()
        self.min_detection_time = 5.0  # Minimum 5 seconds for verification
        
        # Challenge-response system
        self.challenge_active = False
        self.challenge_start_time = 0
        self.challenge_type = None
        self.challenge_completed = False
        
        print("🔒 Advanced Liveness Detection Initialized")
        # print("📋 Required checks: Multiple blinks, mouth movement, head movement, depth analysis")
        print("⏱️  Minimum verification time: 5 seconds")
        
    def eye_aspect_ratio(self, landmarks, eye_points, frame_width, frame_height):
        """Calculate Eye Aspect Ratio for blink detection"""
        pts = [(int(landmarks[p].x * frame_width), int(landmarks[p].y * frame_height)) for p in eye_points]
        vertical1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        vertical2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        horizontal = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    # def mouth_aspect_ratio(self, landmarks, frame_width, frame_height):
    #     """Calculate mouth opening ratio"""
    #     # Get mouth landmarks
    #     mouth_points = []
    #     for landmark_id in MOUTH_LANDMARKS:
    #         point = landmarks[landmark_id]
    #         mouth_points.append([int(point.x * frame_width), int(point.y * frame_height)])
        
    #     mouth_points = np.array(mouth_points)
        
    #     # Calculate mouth opening (vertical distance)
    #     top_lip = np.mean(mouth_points[:6], axis=0)
    #     bottom_lip = np.mean(mouth_points[6:], axis=0)
    #     mouth_opening = np.linalg.norm(top_lip - bottom_lip)
        
    #     return mouth_opening
    
    def calculate_head_pose(self, landmarks, frame_width, frame_height):
        """Calculate head pose using facial landmarks"""
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        points_2d = np.array([
            [nose_tip.x * frame_width, nose_tip.y * frame_height],
            [left_eye.x * frame_width, left_eye.y * frame_height],
            [right_eye.x * frame_width, right_eye.y * frame_height]
        ], dtype=np.float32)
        
        return np.mean(points_2d, axis=0)
    
    def detect_reflection(self, frame, landmarks, frame_width, frame_height):
        """Detect screen reflections that indicate photo/mobile display"""
        # Extract face region
        face_points = []
        key_landmarks = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]
        
        for landmark_id in key_landmarks:
            point = landmarks[landmark_id]
            face_points.append([int(point.x * frame_width), int(point.y * frame_height)])
        
        face_points = np.array(face_points)
        x_min, y_min = np.min(face_points, axis=0)
        x_max, y_max = np.max(face_points, axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame_width, x_max + padding)
        y_max = min(frame_height, y_max + padding)
        
        face_region = frame[y_min:y_max, x_min:x_max]
        if face_region.size == 0:
            return False
        
        # Convert to HSV for better reflection detection
        hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Detect high saturation areas (screen reflections)
        high_sat_mask = cv2.inRange(hsv_face, (0, 100, 100), (179, 255, 255))
        reflection_ratio = np.sum(high_sat_mask > 0) / (face_region.shape[0] * face_region.shape[1])
        
        return reflection_ratio > self.reflection_threshold
    
    def analyze_depth_variation(self, frame, landmarks, frame_width, frame_height):
        """Analyze depth variation - photos are flat, real faces have depth"""
        # Extract multiple facial regions at different depths
        nose_tip = landmarks[1]
        left_cheek = landmarks[116]
        right_cheek = landmarks[345]
        forehead = landmarks[10]
        
        regions = [
            (int(nose_tip.x * frame_width), int(nose_tip.y * frame_height)),
            (int(left_cheek.x * frame_width), int(left_cheek.y * frame_height)),
            (int(right_cheek.x * frame_width), int(right_cheek.y * frame_height)),
            (int(forehead.x * frame_width), int(forehead.y * frame_height))
        ]
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_values = []
        
        for x, y in regions:
            if 0 <= x < frame_width and 0 <= y < frame_height:
                # Extract small region around the point
                region_size = 10
                x_start = max(0, x - region_size)
                y_start = max(0, y - region_size)
                x_end = min(frame_width, x + region_size)
                y_end = min(frame_height, y + region_size)
                
                region = gray_frame[y_start:y_end, x_start:x_end]
                if region.size > 0:
                    # Calculate Laplacian variance (focus measure)
                    laplacian_var = cv2.Laplacian(region, cv2.CV_64F).var()
                    depth_values.append(laplacian_var)
        
        if len(depth_values) < 2:
            return 0
        
        # Calculate variance in depth values
        depth_variance = np.var(depth_values)
        return depth_variance
    
    def calculate_frame_difference(self, current_frame):
        """Calculate frame difference to detect static images"""
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return 0
        
        # Calculate absolute difference between frames
        diff = cv2.absdiff(current_frame, self.prev_frame)
        diff_sum = np.sum(diff)
        
        self.prev_frame = current_frame.copy()
        return diff_sum
    
    def detect_blink(self, landmarks, frame_width, frame_height):
        """Detect multiple blinks for liveness"""
        left_ear = self.eye_aspect_ratio(landmarks, LEFT_EYE, frame_width, frame_height)
        right_ear = self.eye_aspect_ratio(landmarks, RIGHT_EYE, frame_width, frame_height)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < self.ear_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.blink_consec_frames and self.blinks_detected < self.required_blinks:
                self.blinks_detected += 1
                print(f"👁️ Blink {self.blinks_detected}/{self.required_blinks} detected!")
            self.blink_counter = 0
    
    # def detect_mouth_movement(self, landmarks, frame_width, frame_height):
    #     """Detect mouth movements (talking, smiling)"""
    #     mouth_ratio = self.mouth_aspect_ratio(landmarks, frame_width, frame_height)
    #     self.mouth_history.append(mouth_ratio)
        
    #     if len(self.mouth_history) > 20:
    #         self.mouth_history.pop(0)
        
    #     if len(self.mouth_history) >= 10:
    #         mouth_variance = np.var(self.mouth_history)
    #         if mouth_variance > self.mouth_threshold:
    #             if self.mouth_movements < self.required_mouth_movements:
    #                 self.mouth_movements += 1
    #                 print(f"👄 Mouth movement {self.mouth_movements}/{self.required_mouth_movements} detected!")
    #                 self.mouth_history.clear()
    
    def detect_head_movement(self, landmarks, frame_width, frame_height):
        """Detect significant head movements"""
        head_pose = self.calculate_head_pose(landmarks, frame_width, frame_height)
        self.head_pose_history.append(head_pose)
        
        if len(self.head_pose_history) > 15:
            self.head_pose_history.pop(0)
        
        if len(self.head_pose_history) >= 10:
            head_poses = np.array(self.head_pose_history)
            movement_variance = np.var(head_poses, axis=0).sum()
            
            if movement_variance > self.head_movement_threshold:
                if self.head_movements < self.required_head_movements:
                    self.head_movements += 1
                    print(f"🔄 Head movement {self.head_movements}/{self.required_head_movements} detected!")
                    self.head_pose_history = self.head_pose_history[-5:]  # Keep some history
    
    def process_frame(self, frame, landmarks, frame_width, frame_height):
        """Process frame for all liveness checks"""
        # Check for photo/screen reflections
        if self.detect_reflection(frame, landmarks, frame_width, frame_height):
            print("📱 ⚠️ Screen reflection detected! Using a photo or mobile screen.")
            self.is_photo_detected = True
            return False
        
        # Analyze depth variation
        depth_var = self.analyze_depth_variation(frame, landmarks, frame_width, frame_height)
        self.depth_history.append(depth_var)
        
        if len(self.depth_history) > 30:
            self.depth_history.pop(0)
        
        if len(self.depth_history) >= 20:
            avg_depth_var = np.mean(self.depth_history)
            # Only flag as flat image if depth variation is very low consistently
            if avg_depth_var < self.depth_variance_threshold and len(self.depth_history) >= 30:
                # Check if consistently low (not just a few frames)
                low_depth_count = sum(1 for d in self.depth_history[-20:] if d < self.depth_variance_threshold)
                if low_depth_count >= 15:  # 75% of recent frames have low depth
                    print("📄 ⚠️ Low depth variation detected! Possible flat image.")
                    return False
        
        # Check frame differences (static image detection)
        frame_diff = self.calculate_frame_difference(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.frame_diff_history.append(frame_diff)
        
        if len(self.frame_diff_history) > 30:
            self.frame_diff_history.pop(0)
        
        if len(self.frame_diff_history) >= 20:
            avg_frame_diff = np.mean(self.frame_diff_history)
            # Only flag as static if consistently low movement
            if avg_frame_diff < self.min_frame_difference and len(self.frame_diff_history) >= 30:
                low_movement_count = sum(1 for d in self.frame_diff_history[-20:] if d < self.min_frame_difference)
                if low_movement_count >= 15:  # 75% of recent frames have low movement
                    print("🖼️ ⚠️ Static image detected! No natural movement.")
                    return False
        
        # Perform movement detections
        self.detect_blink(landmarks, frame_width, frame_height)
        # self.detect_mouth_movement(landmarks, frame_width, frame_height)
        self.detect_head_movement(landmarks, frame_width, frame_height)
        
        return True
    
    def initiate_challenge(self):
        """Start a challenge-response test"""
        if not self.challenge_active:
            challenges = ["BLINK_TWICE",  "TURN_HEAD_LEFT", "TURN_HEAD_RIGHT"] #"OPEN_MOUTH",
            self.challenge_type = np.random.choice(challenges)
            self.challenge_active = True
            self.challenge_start_time = time.time()
            
            challenge_messages = {
                "BLINK_TWICE": " Please blink twice slowly",
                # "OPEN_MOUTH": "Please open your mouth",
                "TURN_HEAD_LEFT": " Please turn your head to the left",
                "TURN_HEAD_RIGHT": " Please turn your head to the right"
            }
            
            print(f"🎯 Challenge: {challenge_messages.get(self.challenge_type, 'Follow instructions')}")
    
    def is_live_face(self):
        """Comprehensive liveness check"""
        current_time = time.time()
        
        # Check minimum time requirement
        if current_time - self.start_time < self.min_detection_time:
            return False
        
        # Check if photo/screen detected
        if self.is_photo_detected:
            return False
        
        # Check all movement requirements
        blinks_ok = self.blinks_detected >= self.required_blinks
        # mouth_ok = self.mouth_movements >= self.required_mouth_movements
        head_ok = self.head_movements >= self.required_head_movements
        
        # Require depth variation
        depth_ok = len(self.depth_history) >= 20 and np.mean(self.depth_history) >= self.depth_variance_threshold
        
        # Require frame differences
        frame_diff_ok = len(self.frame_diff_history) >= 20 and np.mean(self.frame_diff_history) >= self.min_frame_difference
        
        all_checks_passed = blinks_ok and head_ok and depth_ok and frame_diff_ok # and mouth_ok
        
        if all_checks_passed and not self.challenge_completed:
            self.initiate_challenge()
            # Auto-complete challenge for now (can be enhanced later)
            if self.challenge_active and (current_time - self.challenge_start_time) > 3.0:
                self.challenge_completed = True
                print(f"✅ Challenge completed: {self.challenge_type}")
        
        return all_checks_passed and self.challenge_completed
    
    def get_status(self):
        """Get current status of all checks"""
        current_time = time.time()
        time_remaining = max(0, self.min_detection_time - (current_time - self.start_time))
        
        status = {
            "time_remaining": f"{time_remaining:.1f}s",
            "blinks": f"{self.blinks_detected}/{self.required_blinks}",
            # "mouth": f"{self.mouth_movements}/{self.required_mouth_movements}",
            "head": f"{self.head_movements}/{self.required_head_movements}",
            "depth_ok": len(self.depth_history) >= 20 and np.mean(self.depth_history) >= self.depth_variance_threshold,
            "frame_diff_ok": len(self.frame_diff_history) >= 20 and np.mean(self.frame_diff_history) >= self.min_frame_difference,
            "photo_detected": self.is_photo_detected,
            "challenge_active": self.challenge_active,
            "challenge_type": self.challenge_type
        }
        
        return status
    
    def reset_detection(self):
        """Reset all detection states"""
        self.blinks_detected = 0
        # self.mouth_movements = 0
        self.head_movements = 0
        self.is_photo_detected = False
        self.challenge_active = False
        self.challenge_completed = False
        self.head_pose_history.clear()
        # self.mouth_history.clear()
        self.depth_history.clear()
        self.frame_diff_history.clear()
        self.start_time = time.time()
        print("🔄 Liveness detection reset for next verification")

def create_db_connection():
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS aadhaar_documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  aadhaar_number TEXT, name TEXT, dob TEXT, verification_date TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS passport_documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  passport_number TEXT, name TEXT, date_of_birth TEXT, verification_date TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS licence_documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  dl_number TEXT, name TEXT, dob TEXT, verification_date TIMESTAMP)''')
    return conn

def save_to_database(conn, document_data):
    cursor = conn.cursor()
    doc_type = document_data.get('document_type', '').lower()
    if 'aadhaar' in doc_type:
        cursor.execute("""INSERT INTO aadhaar_documents (aadhaar_number, name, dob, verification_date)
                          VALUES (?, ?, ?, ?)""",
                       (document_data.get('aadhaar_number'), document_data.get('name'),
                        document_data.get('dob'), datetime.now()))
    elif 'passport' in doc_type:
        cursor.execute("""INSERT INTO passport_documents (passport_number, name, date_of_birth, verification_date)
                          VALUES (?, ?, ?, ?)""",
                       (document_data.get('passport_number'), document_data.get('name'),
                        document_data.get('date_of_birth'), datetime.now()))
    elif 'licence' in doc_type or 'license' in doc_type:
        cursor.execute("""INSERT INTO licence_documents (dl_number, name, dob, verification_date)
                          VALUES (?, ?, ?, ?)""",
                       (document_data.get('dl_number'), document_data.get('name'),
                        document_data.get('dob'), datetime.now()))
    conn.commit()

def verify_face():
    conn = create_db_connection()

    # Load cropped face encodings
    cropped_face_encodings, cropped_face_paths, document_data = [], [], {}
    for filename in os.listdir("cropped_faces"):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join("cropped_faces", filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                cropped_face_encodings.append(encoding[0])
                cropped_face_paths.append(image_path)
                doc_path = os.path.join("temp_documents", f"{os.path.splitext(filename)[0]}.json")
                if os.path.exists(doc_path):
                    with open(doc_path, 'r') as f:
                        document_data[image_path] = json.load(f)

    if not cropped_face_encodings:
        print("No cropped faces found for verification")
        return

    print(f"✅ Loaded {len(cropped_face_encodings)} cropped faces")
    print("🔒 Advanced Anti-Spoofing System Active")
    print("📋 Please perform natural movements: blink, speak, move head")
    print("⚠️ Photos and mobile screens will be rejected!")

    video_capture = cv2.VideoCapture(0)
    liveness_detector = AdvancedLivenessDetector()
    face_verified = False
    rejected_count = 0
    max_rejections = 3

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        h, w, _ = frame.shape

        # Face mesh for liveness detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Process frame with comprehensive liveness checks
            frame_valid = liveness_detector.process_frame(frame, landmarks, w, h)
            
            if not frame_valid:
                rejected_count += 1
                if rejected_count >= max_rejections:
                    print("❌ Too many rejections. Please ensure you're using a live webcam, not a photo or mobile screen.")
                    video_capture.release()
                    cv2.destroyAllWindows()
                    conn.close()
                    return
            
            # Check if liveness is confirmed
            if liveness_detector.is_live_face() and not face_verified:
                print("🎉 LIVE FACE CONFIRMED! All anti-spoofing checks passed!")
                print("🔍 Starting face verification...")
                face_verified = True
                
                # Face verification
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(cropped_face_encodings, face_encoding, tolerance=0.5)
                    if True in matches:
                        match_index = matches.index(True)
                        matched_face_path = cropped_face_paths[match_index]
                        doc_data = document_data.get(matched_face_path)
                        if doc_data:
                            save_to_database(conn, doc_data)
                            print(f"✅ VERIFIED! Document data saved: {doc_data.get('name')}")
                            os.remove(matched_face_path)
                            saved_image_path = matched_face_path.replace('cropped_faces', 'saved_images')
                            if os.path.exists(saved_image_path):
                                os.remove(saved_image_path)
                            cropped_face_encodings.pop(match_index)
                            cropped_face_paths.pop(match_index)
                            if not cropped_face_encodings:
                                print("🎉 All faces verified successfully with live detection!")
                                video_capture.release()
                                cv2.destroyAllWindows()
                                conn.close()
                                return
                            else:
                                print(f"📋 {len(cropped_face_encodings)} faces remaining for verification")
                                liveness_detector.reset_detection()
                                face_verified = False
                                rejected_count = 0

        # Display comprehensive status
        status = liveness_detector.get_status()
        y_offset = 30
        
        # Time and basic status
        cv2.putText(frame, f"Time: {status['time_remaining']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Movement checks
        cv2.putText(frame, f"Blinks: {status['blinks']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status['blinks'].split('/')[0] == status['blinks'].split('/')[1] else (0, 0, 255), 2)
        y_offset += 25
        
        # cv2.putText(frame, f"Mouth: {status['mouth']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status['mouth'].split('/')[0] == status['mouth'].split('/')[1] else (0, 0, 255), 2)
        # y_offset += 25
        
        cv2.putText(frame, f"Head: {status['head']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status['head'].split('/')[0] == status['head'].split('/')[1] else (0, 0, 255), 2)
        y_offset += 25
        
        # Anti-spoofing checks
        depth_color = (0, 255, 0) if status['depth_ok'] else (0, 0, 255)
        cv2.putText(frame, f"Depth: {'OK' if status['depth_ok'] else 'FAIL'}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, depth_color, 2)
        y_offset += 25
        
        frame_color = (0, 255, 0) if status['frame_diff_ok'] else (0, 0, 255)
        cv2.putText(frame, f"Movement: {'OK' if status['frame_diff_ok'] else 'FAIL'}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, frame_color, 2)
        y_offset += 25
        
        # Photo detection warning
        if status['photo_detected']:
            cv2.putText(frame, "PHOTO/SCREEN DETECTED!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        # Overall status
        if liveness_detector.is_live_face():
            cv2.putText(frame, "LIVE FACE VERIFIED!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        elif status['photo_detected']:
            cv2.putText(frame, "USE LIVE WEBCAM ONLY!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Perform natural movements", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Challenge instructions
        if status['challenge_active']:
            y_offset += 40
            cv2.putText(frame, f"Challenge: {status['challenge_type']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow('Anti-Spoofing Face Verification System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    verify_face()
