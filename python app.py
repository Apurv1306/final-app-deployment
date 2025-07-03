import glob
import json
import os
import random
import smtplib
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS # Ensure Flask-Cors is imported
import base64

# --- Configuration Constants (from original Kivy app) ---
SAMPLES_PER_USER: int = 10
FRAME_REDUCE_FACTOR: float = 0.5 # Not directly used for backend processing, but good to keep in mind for frame quality
RECOGNITION_INTERVAL: int = 5 * 60 # 5 minutes
AUDIO_FILE: str = "thank_you.mp3" # This will be handled by frontend
TICK_ICON_PATH: str = "tick.png" # This will be handled by frontend
HAAR_CASCADE_PATH: str = "./haarcascade_frontalface_default.xml" # Ensure this path is correct relative to app.py

GOOGLE_FORM_VIEW_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/viewform"
)
GOOGLE_FORM_POST_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"
)
FORM_FIELDS: Dict[str, str] = {
    "name": "entry.935510406",
    "emp_id": "entry.886652582",
    "date": "entry.1160275796",
    "time": "entry.32017675",
}

# Environment variables for sensitive info
# These will be passed via buildozer.spec's android.env_vars
EMAIL_ADDRESS: str = os.environ.get("FACEAPP_EMAIL", "faceapp0011@gmail.com")
EMAIL_PASSWORD: str = os.environ.get("FACEAPP_PASS", "ytup bjrd pupf tuuj") # Use an app-specific password for Gmail
SMTP_SERVER: str = "smtp.gmail.com"
SMTP_PORT: int = 587
ADMIN_EMAIL_ADDRESS: str = os.environ.get("FACEAPP_ADMIN_EMAIL", "projects@archtechautomation.com")

# Simple logger for backend console output
def Logger(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def python_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _crop_and_resize_for_passport(cv_image: np.ndarray, target_size: Tuple[int, int] = (240, 320)) -> np.ndarray:
    """
    Crops and resizes an image to a target aspect ratio and size,
    similar to passport photo requirements.
    """
    h, w = cv_image.shape[:2]
    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height
    current_aspect_ratio = w / h

    cropped_image = cv_image
    if current_aspect_ratio > target_aspect_ratio:
        new_width = int(h * target_aspect_ratio)
        x_start = (w - new_width) // 2
        cropped_image = cv_image[:, x_start : x_start + new_width]
    elif current_aspect_ratio < target_aspect_ratio:
        new_height = int(w / target_aspect_ratio)
        y_start = (h - new_height) // 2
        cropped_image = cv_image[y_start : y_start + new_height, :]

    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

class FaceAppBackend:
    def __init__(self):
        # Determine the base directory for app data on Android
        # ANDROID_APP_PATH is set by python-for-android to the app's private data directory
        app_data_base_dir = os.environ.get("ANDROID_APP_PATH", ".")
        self.known_faces_dir: str = str(Path(app_data_base_dir) / "known_faces")
        ensure_dir(self.known_faces_dir) # Ensure the directory exists
        Logger(f"[INFO] Known faces directory set to: {self.known_faces_dir}")

        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            Logger(f"[WARN] Failed to load Haar cascade from '{HAAR_CASCADE_PATH}'. Attempting fallback to OpenCV data path.")
            fallback_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(fallback_path)
            if self.face_cascade.empty():
                error_msg = (f"[ERROR] Failed to load Haar cascade classifier from both "
                             f"'{HAAR_CASCADE_PATH}' and '{fallback_path}'. "
                             f"Please ensure 'haarcascade_frontalface_default.xml' is present and accessible "
                             f"in your project folder.")
                Logger(error_msg)
                raise RuntimeError(error_msg)
            else:
                Logger(f"[INFO] Successfully loaded Haar cascade from fallback path: '{fallback_path}'.")
        else:
            Logger(f"[INFO] Successfully loaded Haar cascade from: '{HAAR_CASCADE_PATH}'.")

        self.recognizer = None
        self.label_map = {}
        self.last_seen_time: Dict[str, float] = {}
        self.otp_storage: Dict[str, str] = {}
        self.pending_names: Dict[str, Optional[str]] = {} # Stores name for capture process
        self.user_emails: Dict[str, str] = {}
        self.last_recognized_info: Optional[Dict[str, Any]] = None # Initialize for frontend polling

        self.capture_mode: bool = False # Flag to indicate if samples are being captured
        self.capture_target_count: int = 0
        self.capture_collected_count: int = 0
        self.capture_name: Optional[str] = None
        self.capture_emp_id: Optional[str] = None
        self.capture_start_index: int = 0
        self.capture_lock = threading.Lock() # To prevent race conditions during capture

        self._train_recognizer_and_load_emails() # Initial training and email loading

    def _train_recognizer_and_load_emails(self):
        """Initializes recognizer and loads user emails."""
        self.recognizer, self.label_map = self._train_recognizer()
        self.user_emails = self._load_emails()

    def _train_recognizer(self):
        """
        Trains the LBPHFaceRecognizer with images found in known_faces_dir.
        Returns the trained recognizer and a label map.
        """
        images: list[np.ndarray] = []
        labels: list[int] = []
        label_map: Dict[int, Tuple[str, str]] = {}
        label_id = 0

        ensure_dir(self.known_faces_dir) # Ensure the directory exists before listing
        for file in sorted(os.listdir(self.known_faces_dir)):
            if not file.lower().endswith((".jpg", ".png")):
                continue
            try:
                # Filename format: name_emp_id_XXX.jpg
                parts = file.split("_")
                if len(parts) < 3:
                    Logger(f"[WARN] Skipping unrecognised filename format: {file}")
                    continue
                # Reconstruct name if it had underscores, assuming last two parts are emp_id and index
                name = "_".join(parts[:-2]).lower()
                emp_id = parts[-2].upper()
            except ValueError:
                Logger(f"[WARN] Skipping unrecognised filename format: {file}")
                continue

            img_path = Path(self.known_faces_dir) / file
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                Logger(f"[WARN] Could not read image: {img_path}")
                continue

            img_resized = cv2.resize(img_gray, (200, 200)) # Standardize size for training
            images.append(img_resized)

            # Assign a unique label_id for each unique (name, emp_id) pair
            current_identity = (name, emp_id)
            if current_identity not in label_map.values():
                label_map[label_id] = current_identity
                labels.append(label_id)
                label_id += 1
            else:
                # Find existing label_id for this identity
                existing_label_id = -1
                for lbl, identity in label_map.items():
                    if identity == current_identity:
                        existing_label_id = lbl
                        break
                labels.append(existing_label_id)

        recogniser = cv2.face.LBPHFaceRecognizer_create()
        if images:
            try:
                recogniser.train(images, np.array(labels))
                Logger(f"[INFO] Trained recogniser on {len(images)} images across {len(label_map)} identities.")
            except cv2.error as e:
                Logger(f"[ERROR] OpenCV training error: {e}. This might happen if there's only one sample or all samples are identical.")
                recogniser = None # Mark as untrained
        else:
            Logger("[INFO] No images found – recogniser disabled until first registration.")
            recogniser = None # Mark as untrained

        return recogniser, label_map

    def _load_emails(self) -> Dict[str, str]:
        """Loads user emails from a JSON file."""
        emails_file = Path(self.known_faces_dir) / "user_emails.json"
        if emails_file.is_file():
            try:
                with emails_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                Logger(f"[WARN] Invalid JSON in email storage: {exc}; starting fresh.")
        return {}

    def _save_email(self, emp_id: str, email: str) -> None:
        """Saves a user's email to the JSON file."""
        self.user_emails[emp_id] = email
        with (Path(self.known_faces_dir) / "user_emails.json").open("w", encoding="utf-8") as f:
            json.dump(self.user_emails, f, indent=2)

    def _generate_otp(self) -> str:
        """Generates a 6-digit OTP."""
        return str(random.randint(100000, 999999))

    def _send_otp_email(self, email: str, otp: str, name: str, emp_id: str, is_admin_email: bool = False) -> bool:
        """Sends an OTP or admin notification email."""
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = email
        if is_admin_email:
            msg["Subject"] = f"FaceApp Notification: Person Details - {name.title()} ({emp_id})"
            body_html = (
                f"<p>Details of a person for whom an OTP process was initiated:</p>"
                f"<p><b>Name:</b> {name.title()}<br>"
                f"<b>Employee ID:</b> {emp_id}</p>"
                f"<p>Generated OTP: <b>{otp}</b></p>"
            )
        else:
            msg["Subject"] = "Your FaceApp OTP"
            body_html = (
                f"<h2>OTP Verification for {name.title()} ({emp_id})</h2><p>Your OTP is <b>{otp}</b>. "
                "It is valid for 10 minutes.</p>"
                "<p>Please use this OTP to proceed with your photo update/registration.</p>"
            )
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            Logger(f"[INFO] Sent {'admin notification' if is_admin_email else 'OTP'} to {email}")
            return True
        except Exception as exc:
            Logger(f"[ERROR] SMTP error when sending {'admin notification' if is_admin_email else 'OTP'} to {email}: {exc}")
            return False

    def _submit_to_google_form(self, name: str, emp_id: str) -> None:
        """Submits attendance data to a Google Form."""
        payload = {
            FORM_FIELDS["name"]: name.title(),
            FORM_FIELDS["emp_id"]: emp_id,
            FORM_FIELDS["date"]: datetime.now().strftime("%d/%m/%Y"),
            FORM_FIELDS["time"]: datetime.now().strftime("%H:%M:%S"),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (FaceApp Attendance Bot)",
            "Referer": GOOGLE_FORM_VIEW_URL,
        }
        Logger(f"[INFO] Attempting to submit attendance for {name} ({emp_id}) to URL: {GOOGLE_FORM_POST_URL}")
        Logger(f"[INFO] Payload: {payload}")
        try:
            with requests.Session() as session:
                resp = session.post(GOOGLE_FORM_POST_URL, data=payload, headers=headers, timeout=10, allow_redirects=False)
            if resp.status_code in (200, 302):
                Logger("[INFO] Attendance submitted successfully to Google Form.")
                # Frontend will display success message
            else:
                Logger(f"[WARN] Google Form submission returned status {resp.status_code}. Response: {resp.text[:200]}...")
                # Frontend will display warning
        except requests.exceptions.Timeout:
            Logger(f"[ERROR] Google Form submission timed out for {name} ({emp_id}).")
            # Frontend will display error
        except requests.exceptions.ConnectionError as exc:
            Logger(f"[ERROR] Google Form submission connection error for {name} ({emp_id}): {exc}")
            # Frontend will display error
        except requests.RequestException as exc:
            Logger(f"[ERROR] An unexpected error occurred during form submission for {name} ({emp_id}): {exc}")
            # Frontend will display error

    def process_frame(self, frame_data_b64: str) -> Dict[str, Any]:
        """
        Processes a single frame for face detection and recognition.
        If in capture mode, it saves face samples.
        Returns detection/recognition results.
        """
        try:
            # Decode base64 image
            nparr = np.frombuffer(base64.b64decode(frame_data_b64), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                Logger("[ERROR] Could not decode image data.")
                return {"status": "error", "message": "Invalid image data"}

            h, w = frame.shape[:2]
            # Reduce frame size for faster processing, similar to Kivy app's FRAME_REDUCE_FACTOR
            # This is applied here as the frontend might send full resolution frames.
            # If frontend already resizes, this can be removed or adjusted.
            resized = cv2.resize(
                frame, (int(w * 0.5), int(h * 0.5))
            )
            gray_small = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            try:
                faces = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)
            except cv2.error as e:
                Logger(f"[ERROR] OpenCV error in detectMultiScale: {e}")
                faces = []

            results = []
            for (x, y, w_s, h_s) in faces:
                # Scale back to original frame coordinates
                x_full, y_full, w_full, h_full = [
                    int(v / 0.5) for v in (x, y, w_s, h_s)
                ]

                # Extract face ROI from original frame for better quality recognition/storage
                expansion_factor = 1.8
                exp_w = int(w_full * expansion_factor)
                exp_h = int(h_full * expansion_factor)
                center_x = x_full + w_full // 2
                center_y = y_full + h_full // 2
                exp_x = center_x - exp_w // 2
                exp_y = center_y - exp_h // 2

                frame_h, frame_w = frame.shape[:2]
                exp_x = max(0, min(exp_x, frame_w - exp_w))
                exp_y = max(0, min(exp_y, frame_h - exp_h))
                exp_w = min(exp_w, frame_w - exp_x)
                exp_h = min(exp_h, frame_h - exp_y)

                # Ensure valid ROI dimensions before slicing
                if exp_w <= 0 or exp_h <= 0:
                    Logger(f"[WARN] Invalid face ROI dimensions: w={exp_w}, h={exp_h}. Skipping.")
                    continue

                color_face_roi = frame[exp_y : exp_y + exp_h, exp_x : exp_x + exp_w].copy()
                grayscale_face_roi = cv2.cvtColor(color_face_roi, cv2.COLOR_BGR2GRAY)

                name = "unknown"
                emp_id = ""
                conf = 1000 # High confidence for unknown

                if self.recognizer:
                    try:
                        label, conf = self.recognizer.predict(grayscale_face_roi)
                        name, emp_id = self.label_map.get(label, ("unknown", ""))
                    except Exception as e:
                        Logger(f"[ERROR] Recognizer prediction failed: {e}")
                        label, conf = -1, 1000 # Reset to unknown on error

                face_info = {
                    "box": [x_full, y_full, w_full, h_full],
                    "name": name.title(),
                    "emp_id": emp_id,
                    "confidence": float(conf),
                    "status": "unknown"
                }

                if self.capture_mode:
                    with self.capture_lock:
                        if self.capture_collected_count < self.capture_target_count:
                            face_img_resized = cv2.resize(grayscale_face_roi, (200, 200))
                            filename = f"{self.capture_name}_{self.capture_emp_id}_{self.capture_start_index + self.capture_collected_count:03d}.jpg"
                            cv2.imwrite(str(Path(self.known_faces_dir) / filename), face_img_resized)
                            self.capture_collected_count += 1
                            Logger(f"[INFO] Captured sample {self.capture_collected_count}/{self.capture_target_count} for {self.capture_emp_id}")
                            face_info["capture_progress"] = f"{self.capture_collected_count}/{self.capture_target_count}"
                            face_info["status"] = "capturing"
                            if self.capture_collected_count >= self.capture_target_count:
                                Logger("[INFO] Capture complete – retraining recogniser…")
                                self.capture_mode = False # End capture mode
                                # Retrain recognizer in a separate thread to avoid blocking
                                threading.Thread(target=self._retrain_after_capture, daemon=True).start()
                                face_info["status"] = "capture_complete"
                        else:
                            face_info["status"] = "capturing" # Still in capture mode until retraining is done
                else: # Normal recognition mode
                    if conf < 60: # Threshold for known face
                        now = time.time()
                        last_seen = self.last_seen_time.get(emp_id, 0)
                        if now - last_seen > RECOGNITION_INTERVAL:
                            self.last_seen_time[emp_id] = now
                            face_info["status"] = "recognized_new"
                            # Trigger attendance submission and preview update in separate threads
                            threading.Thread(
                                target=self._handle_successful_recognition,
                                args=(name, emp_id, color_face_roi),
                                daemon=True,
                                name="AttendanceSubmitter",
                            ).start()
                        else:
                            face_info["status"] = "recognized_recent"
                    else:
                        face_info["status"] = "unknown"

                results.append(face_info)

            return {"status": "success", "faces": results}

        except Exception as e:
            Logger(f"[ERROR] Error processing frame: {e}")
            return {"status": "error", "message": str(e)}

    def _retrain_after_capture(self):
        """Retrains the recognizer after samples are captured."""
        self.recognizer, self.label_map = self._train_recognizer()
        Logger("[INFO] Recognizer retraining finished.")


    def _handle_successful_recognition(self, name: str, emp_id: str, face_roi_color: np.ndarray):
        """Handles post-recognition actions like attendance submission."""
        Logger(f"[INFO] Recognised {name} ({emp_id}) – submitting attendance…")
        # Process face image for display on frontend
        processed_face_image = _crop_and_resize_for_passport(face_roi_color, (240, 320))
        _, buffer = cv2.imencode('.jpg', processed_face_image)
        face_image_b64 = base64.b64encode(buffer).decode('utf-8')

        current_time = datetime.now().strftime("%H:%M:%S")

        # Frontend will receive this info via a separate mechanism or poll
        # For now, just submit to Google Form
        threading.Thread(target=self._submit_to_google_form, args=(name, emp_id), daemon=True, name="GoogleFormSubmitter").start()

        # Store last recognized info, frontend can poll this or receive via WebSocket if implemented
        self.last_recognized_info = {
            "name": name.title(),
            "emp_id": emp_id,
            "time": current_time,
            "image": face_image_b64
        }

    def start_capture_samples(self, name: str, emp_id: str, updating: bool = False, sample_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Initiates the sample capture process.
        Sets internal flags for `process_frame` to start saving images.
        """
        with self.capture_lock:
            if self.capture_mode:
                return {"status": "error", "message": "Already in capture mode."}

            resolved_name = name or next((nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id), None)
            if resolved_name is None:
                return {"status": "error", "message": "No existing face found for this ID – please register first or provide a name."}

            self.capture_name = resolved_name
            self.capture_emp_id = emp_id
            self.capture_target_count = sample_count if sample_count else SAMPLES_PER_USER
            self.capture_collected_count = 0

            pattern = str(Path(self.known_faces_dir) / f"{self.capture_name}_{self.capture_emp_id}_*.jpg")
            existing_files = glob.glob(pattern)
            self.capture_start_index = len(existing_files)

            self.capture_mode = True
            Logger(f"[INFO] Backend starting sample capture for {emp_id} – target {self.capture_target_count} faces (updating={updating}).")
            return {"status": "success", "message": "Capture mode initiated."}

    def stop_capture_samples(self) -> Dict[str, Any]:
        """Stops the sample capture process."""
        with self.capture_lock:
            if not self.capture_mode:
                return {"status": "error", "message": "Not in capture mode."}
            self.capture_mode = False
            Logger("[INFO] Backend stopping sample capture.")
            return {"status": "success", "message": "Capture mode stopped."}

    def get_user_email(self, emp_id: str) -> Dict[str, Any]:
        """Retrieves email for a given employee ID."""
        email = self.user_emails.get(emp_id)
        name = next((nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id), None)
        return {"status": "success", "email": email, "name": name}

    def send_otp_flow(self, emp_id: str, email: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Initiates the OTP sending process."""
        resolved_name = name or next((nm for _lbl, (nm, eid) in self.label_map.items() if eid == emp_id), "Unknown User")
        otp = self._generate_otp()
        self.otp_storage[emp_id] = otp
        self.pending_names[emp_id] = resolved_name # Store name for later capture

        # Send emails in a separate thread
        def _send_thread():
            user_mail_ok = self._send_otp_email(email, otp, resolved_name, emp_id, False)
            admin_mail_ok = self._send_otp_email(ADMIN_EMAIL_ADDRESS, otp, resolved_name, emp_id, True)
            if not user_mail_ok:
                Logger(f"[WARN] Failed to send OTP email to user {email}.")
            if not admin_mail_ok:
                Logger(f"[WARN] Failed to send admin notification email to {ADMIN_EMAIL_ADDRESS}.")

        threading.Thread(target=_send_thread, daemon=True, name="SendOTPThread").start()
        return {"status": "success", "message": "OTP sending initiated."}

    def verify_otp(self, emp_id: str, otp_entered: str) -> Dict[str, Any]:
        """Verifies the entered OTP."""
        if self.otp_storage.get(emp_id) == otp_entered:
            # OTP is correct, clear it and return success
            del self.otp_storage[emp_id]
            return {"status": "success", "message": "OTP verified successfully."}
        else:
            return {"status": "error", "message": "Incorrect OTP."}

    def register_user_email(self, emp_id: str, email: str) -> Dict[str, Any]:
        """Registers a user's email."""
        self._save_email(emp_id, email)
        return {"status": "success", "message": "Email registered."}

    def get_last_recognized_info(self) -> Dict[str, Any]:
        """Returns the last recognized person's info for frontend display."""
        info = getattr(self, 'last_recognized_info', None)
        if info:
            # Clear after sending to ensure new recognition triggers update
            self.last_recognized_info = None
            return {"status": "success", "info": info}
        return {"status": "no_new_info"}

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# Initialize the backend logic
face_app_backend = FaceAppBackend()

@app.route('/')
def index_route(): # Renamed to avoid conflict with index.html filename if Python app.py is run directly
    # For the webview bootstrap, this route might not be directly accessed by the user.
    # The webview will load 'http://127.0.0.1:5000' which is the base of your Flask app.
    # You could serve index.html here if you put it in a 'templates' folder.
    # For now, it just confirms the backend is running.
    return "FaceApp Backend is running."

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"status": "error", "message": "No image data provided"}), 400

    frame_data_b64 = data['image'].split(',')[1] # Remove "data:image/jpeg;base64," prefix
    result = face_app_backend.process_frame(frame_data_b64)
    return jsonify(result)

@app.route('/register_user', methods=['POST'])
def register_user_endpoint():
    data = request.json
    name = data.get('name')
    emp_id = data.get('emp_id')
    email = data.get('email')

    if not all([name, emp_id, email]):
        return jsonify({"status": "error", "message": "Missing name, employee ID, or email"}), 400
    if "@" not in email:
        return jsonify({"status": "error", "message": "Invalid email format"}), 400

    face_app_backend.register_user_email(emp_id, email) # Save email immediately
    # Start capture for registration (SAMPLES_PER_USER)
    result = face_app_backend.start_capture_samples(name, emp_id, updating=False, sample_count=SAMPLES_PER_USER)
    return jsonify(result)

@app.route('/get_user_email', methods=['POST'])
def get_user_email_endpoint():
    data = request.json
    emp_id = data.get('emp_id')
    if not emp_id:
        return jsonify({"status": "error", "message": "Missing employee ID"}), 400
    result = face_app_backend.get_user_email(emp_id)
    return jsonify(result)

@app.route('/send_otp', methods=['POST'])
def send_otp_endpoint():
    data = request.json
    emp_id = data.get('emp_id')
    email = data.get('email')
    name = data.get('name') # Optional, for admin email context

    if not all([emp_id, email]):
        return jsonify({"status": "error", "message": "Missing employee ID or email"}), 400

    result = face_app_backend.send_otp_flow(emp_id, email, name)
    return jsonify(result)

@app.route('/verify_otp', methods=['POST'])
def verify_otp_endpoint():
    data = request.json
    emp_id = data.get('emp_id')
    otp = data.get('otp')

    if not all([emp_id, otp]):
        return jsonify({"status": "error", "message": "Missing employee ID or OTP"}), 400

    result = face_app_backend.verify_otp(emp_id, otp)
    return jsonify(result)

@app.route('/start_update_capture', methods=['POST'])
def start_update_capture_endpoint():
    data = request.json
    name = data.get('name')
    emp_id = data.get('emp_id')
    # Use a smaller sample count for updates, e.g., 5
    result = face_app_backend.start_capture_samples(name, emp_id, updating=True, sample_count=5)
    return jsonify(result)

@app.route('/get_last_recognized', methods=['GET'])
def get_last_recognized_endpoint():
    """Endpoint for frontend to poll for last recognized person's details."""
    result = face_app_backend.get_last_recognized_info()
    return jsonify(result)

if __name__ == '__main__':
    # To run this:
    # 1. pip install Flask opencv-python numpy requests Flask-Cors
    # 2. Download haarcascade_frontalface_default.xml and place it in the same directory as app.py
    #    (You can find it in OpenCV's data directory, e.g., site-packages/cv2/data/)
    # 3. Set environment variables: FACEAPP_EMAIL, FACEAPP_PASS, FACEAPP_ADMIN_EMAIL
    #    Example (Linux/macOS):
    #    export FACEAPP_EMAIL="your_email@gmail.com"
    #    export FACEAPP_PASS="your_app_password" # Use app password for Gmail, not your main password
    #    export FACEAPP_ADMIN_EMAIL="admin_email@example.com"
    # 4. python "python app.py"
    app.run(host='0.0.0.0', port=5000, debug=True)
