"""Real-Time Driver Drowsiness & Yawn Alert System — Telemetry HUD Edition.

Tracks 68 facial landmarks using OpenCV and dlib, computing Eye Aspect Ratio (EAR)
and Mouth Aspect Ratio (MAR) via SciPy with a translucent heads-up display (HUD)
and low-latency pygame audio alerts.
"""

import math
import os
import struct
import sys
import time
import wave

import cv2
import dlib
import numpy as np
import pygame
from scipy.spatial import distance as dist

from setup_model import MODEL_FILENAME, ensure_landmark_model

# Detection thresholds and consecutive frame limits
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 18
MAR_THRESHOLD = 0.70
MAR_CONSEC_FRAMES = 14

ALARM_PATH = "alarm.wav"


def ensure_alarm_wav(filepath: str = ALARM_PATH) -> None:
    """Generate a fallback two-tone alert sound if alarm.wav is missing."""
    if os.path.exists(filepath):
        return
    sample_rate = 44100
    duration = 0.6
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(int(sample_rate * duration)):
            t = i / sample_rate
            freq = 980.0 if t < 0.3 else 1320.0
            sample = int(32767 * 0.6 * math.sin(2 * math.pi * freq * t))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """Compute Eye Aspect Ratio (EAR) from 6 eye landmark points."""
    vertical_a = dist.euclidean(eye[1], eye[5])
    vertical_b = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])
    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """Compute Mouth Aspect Ratio (MAR) from 12 outer mouth landmark points."""
    vertical_a = dist.euclidean(mouth[2], mouth[10])
    vertical_b = dist.euclidean(mouth[3], mouth[9])
    vertical_c = dist.euclidean(mouth[4], mouth[8])
    horizontal = dist.euclidean(mouth[0], mouth[6])
    return float((vertical_a + vertical_b + vertical_c) / (2.0 * horizontal))


def shape_to_np(shape: dlib.full_object_detection) -> np.ndarray:  # type: ignore
    """Convert dlib full_object_detection landmarks into a (68, 2) NumPy array."""
    coords = np.zeros((68, 2), dtype="int")
    for idx in range(68):
        coords[idx] = (shape.part(idx).x, shape.part(idx).y)
    return coords


def configure_low_latency(cap: cv2.VideoCapture) -> None:
    """Set 640x480 resolution and minimal buffer size to eliminate stream lag."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def discover_cameras(max_indices: int = 5) -> list[int]:
    """Scan and return all working camera indices."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    working = []
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                working.append(idx)
        cap.release()
    return working


def open_camera_by_index(index: int) -> cv2.VideoCapture:
    """Open a specific camera index with low-latency capture settings."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")
    configure_low_latency(cap)
    return cap


def draw_translucent_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.65,
) -> None:
    """Draw a semi-transparent filled rectangle onto the video frame."""
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(frame.shape[1], pt2[0]), min(frame.shape[0], pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame[y1:y2, x1:x2].copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), color, -1)
    cv2.addWeighted(
        overlay, alpha, frame[y1:y2, x1:x2], 1.0 - alpha, 0, frame[y1:y2, x1:x2]
    )


def draw_corner_reticle(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
    length: int = 18,
    thickness: int = 2,
) -> None:
    """Draw HUD corner brackets around the detected face region."""
    # Top-left
    cv2.line(frame, (x, y), (x + length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + length), color, thickness)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)


def draw_gauge_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    max_val: float,
    threshold: float,
    bar_color: tuple[int, int, int],
    label: str,
) -> None:
    """Render a labeled HUD progress bar with a vertical threshold marker."""
    cv2.putText(
        frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 240), 1
    )
    cv2.rectangle(frame, (x, y), (x + width, y + height), (55, 65, 75), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (110, 125, 140), 1)

    fill_ratio = max(0.0, min(1.0, value / max_val))
    fill_w = int(width * fill_ratio)
    if fill_w > 0:
        cv2.rectangle(
            frame, (x + 1, y + 1), (x + fill_w, y + height - 1), bar_color, -1
        )

    thresh_x = x + int(width * max(0.0, min(1.0, threshold / max_val)))
    cv2.line(frame, (thresh_x, y - 3), (thresh_x, y + height + 3), (255, 255, 255), 2)


def render_hud(
    frame: np.ndarray,
    status_text: str,
    status_color: tuple[int, int, int],
    ear: float,
    mar: float,
    fps: float,
    source_label: str,
    face_detected: bool,
    multi_cam: bool,
) -> None:
    """Composite the top status bar, bottom gauge dock, and center alert banner."""
    h, w = frame.shape[:2]

    # Top HUD header bar
    draw_translucent_rect(frame, (0, 0), (w, 46), (14, 18, 24), alpha=0.72)
    cv2.line(frame, (0, 46), (w, 46), status_color, 2)

    cv2.circle(frame, (20, 23), 6, status_color, -1)
    cv2.putText(
        frame,
        f"DRIVER HUD // {status_text}",
        (36, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        status_color,
        2,
    )

    telemetry_right = f"{source_label} | {fps:.1f} FPS"
    text_size = cv2.getTextSize(telemetry_right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    cv2.putText(
        frame,
        telemetry_right,
        (w - text_size[0] - 16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 235, 255),
        2,
    )

    # Center warning banner during active alerts
    if "ALERT" in status_text:
        banner_w = 340
        bx1 = (w - banner_w) // 2
        draw_translucent_rect(
            frame, (bx1, 60), (bx1 + banner_w, 102), (0, 0, 180), alpha=0.78
        )
        cv2.rectangle(frame, (bx1, 60), (bx1 + banner_w, 102), status_color, 2)
        cv2.putText(
            frame,
            f"! {status_text} !",
            (bx1 + 38, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )

    # Bottom HUD telemetry dock
    dock_top = h - 68
    draw_translucent_rect(frame, (0, dock_top), (w, h), (14, 18, 24), alpha=0.72)
    cv2.line(frame, (0, dock_top), (w, dock_top), (70, 90, 110), 1)

    bar_w = max(130, (w - 90) // 2)
    ear_color = (
        (0, 60, 255) if (face_detected and ear < EAR_THRESHOLD) else (0, 230, 120)
    )
    mar_color = (
        (0, 150, 255) if (face_detected and mar > MAR_THRESHOLD) else (255, 200, 0)
    )

    ear_label = (
        f"EYE ASPECT RATIO (EAR): {ear:.2f}"
        if face_detected
        else "EYE ASPECT RATIO: --"
    )
    mar_label = (
        f"MOUTH ASPECT RATIO (MAR): {mar:.2f}"
        if face_detected
        else "MOUTH ASPECT RATIO: --"
    )

    draw_gauge_bar(
        frame,
        18,
        dock_top + 26,
        bar_w,
        14,
        ear if face_detected else 0.0,
        0.45,
        EAR_THRESHOLD,
        ear_color,
        ear_label,
    )
    draw_gauge_bar(
        frame,
        36 + bar_w,
        dock_top + 26,
        bar_w,
        14,
        mar if face_detected else 0.0,
        1.00,
        MAR_THRESHOLD,
        mar_color,
        mar_label,
    )

    hint = "[C] Switch Cam   [Q] Quit" if multi_cam else "[Q] Quit"
    hint_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    cv2.putText(
        frame,
        hint,
        (w - hint_size[0] - 16, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (175, 190, 205),
        1,
    )


def main() -> None:
    model_path = ensure_landmark_model(MODEL_FILENAME)
    ensure_alarm_wav(ALARM_PATH)

    pygame.mixer.init()
    alarm_sound = pygame.mixer.Sound(ALARM_PATH)

    detector = dlib.get_frontal_face_detector()  # type: ignore
    predictor = dlib.shape_predictor(model_path)  # type: ignore

    custom_source = sys.argv[1] if len(sys.argv) > 1 else None
    available_cams: list[int] = []
    cam_pos = 0

    if custom_source is not None:
        if custom_source.isdigit():
            idx = int(custom_source)
            available_cams = [idx]
            cap = open_camera_by_index(idx)
            source_label = f"CAM {idx}"
        else:
            cap = cv2.VideoCapture(custom_source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open stream/video: {custom_source}")
            configure_low_latency(cap)
            source_label = "IP STREAM"
    else:
        available_cams = discover_cameras()
        if not available_cams:
            raise RuntimeError(
                "No working cameras found.\n"
                "1. Ensure Iriun/DroidCam is open and streaming from your phone.\n"
                "2. Check Windows Settings -> Privacy & security -> Camera -> 'Let desktop apps access your camera'."
            )
        # Prefer index 1+ (phone/external webcam) over index 0 (built-in laptop webcam)
        cam_pos = 1 if len(available_cams) > 1 else 0
        active_idx = available_cams[cam_pos]
        cap = open_camera_by_index(active_idx)
        source_label = f"CAM {active_idx}"
        print(f"[INFO] Available cameras: {available_cams} | Active: {source_label}")

    eye_counter = 0
    yawn_counter = 0
    prev_time = time.time()

    print("[INFO] Driver HUD active. Press 'c' to switch camera | Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video stream ended.")
                break

            # Resize large streams to 640px wide for fast, lag-free dlib inference
            if frame.shape[1] > 640:
                scale = 640.0 / frame.shape[1]
                frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            status_text = "MONITORING"
            status_color = (0, 230, 120)
            ear = 0.0
            mar = 0.0
            face_detected = len(faces) > 0

            for face in faces:
                fx, fy, fw, fh = face.left(), face.top(), face.width(), face.height()
                landmarks = shape_to_np(predictor(gray, face))

                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                mouth = landmarks[48:60]

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                mar = mouth_aspect_ratio(mouth)

                cv2.drawContours(
                    frame, [cv2.convexHull(left_eye)], -1, (0, 235, 255), 1
                )
                cv2.drawContours(
                    frame, [cv2.convexHull(right_eye)], -1, (0, 235, 255), 1
                )
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 200, 0), 1)

                if ear < EAR_THRESHOLD:
                    eye_counter += 1
                    if eye_counter >= EAR_CONSEC_FRAMES:
                        status_text = "DROWSINESS ALERT"
                        status_color = (0, 50, 255)
                        if not pygame.mixer.get_busy():
                            alarm_sound.play()
                else:
                    eye_counter = 0

                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= MAR_CONSEC_FRAMES:
                        status_text = "YAWN ALERT"
                        status_color = (0, 145, 255)
                        if not pygame.mixer.get_busy():
                            alarm_sound.play()
                else:
                    yawn_counter = 0

                draw_corner_reticle(frame, fx, fy, fw, fh, status_color)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            render_hud(
                frame=frame,
                status_text=status_text,
                status_color=status_color,
                ear=ear,
                mar=mar,
                fps=fps,
                source_label=source_label,
                face_detected=face_detected,
                multi_cam=len(available_cams) > 1,
            )

            cv2.imshow("Driver Drowsiness & Yawn Alert System", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c") and len(available_cams) > 1:
                cap.release()
                cam_pos = (cam_pos + 1) % len(available_cams)
                active_idx = available_cams[cam_pos]
                cap = open_camera_by_index(active_idx)
                source_label = f"CAM {active_idx}"
                eye_counter = 0
                yawn_counter = 0
                print(f"[INFO] Switched to {source_label}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
