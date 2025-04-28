import os
import cv2
import time
import csv
import base64
import serial
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio

# === Paths & Configs ===
base_dir = "details"
images_dir = os.path.join(base_dir, "images")
original_dir = os.path.join(images_dir, "original")
cropped_dir = os.path.join(images_dir, "cropped")
reduced_dir = os.path.join(images_dir, "reduced")
csv_dir = os.path.join(base_dir, "csv")
log_file = os.path.join(csv_dir, "baby_monitoring_data.csv")

model_path = "best_ncnn_model"
audio_model_path = "yamnet.tflite"

# === Audio Configs ===
score_threshold = 0.3
max_results = 5
cry_detected_flag = False
global_timestamp_ms = 0

# === LoRa Config ===
serial_port = "/dev/ttyUSB0"
baud_rate = 115200
chunk_size = 240
HEARTBEAT_INTERVAL = 30
lora = serial.Serial(serial_port, baud_rate)
time.sleep(2)

# === Initial Setup ===
if os.path.exists(base_dir):
    import shutil
    shutil.rmtree(base_dir)

os.makedirs(original_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(reduced_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

with open(log_file, "w", newline="") as f:
    csv.writer(f).writerow(["Timestamp", "Baby Detection", "Cry Detection", "Cropped Image"])

# === Utility Functions ===
def get_timestamp():
    return time.strftime("%Y-%m-%d_%H%M%S")

def save_compressed_image(img, path, quality=90):
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode(".jpg", img, encode_param)
    if success:
        with open(path, "wb") as f:
            f.write(encoded.tobytes())
    else:
        print("Failed to encode image.")

def compress_to_96x96_under_2kb(img, save_path=None, max_kb=2):
    resized = cv2.resize(img, (96, 96))
    quality = 90
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    while quality >= 10:
        success, encoded = cv2.imencode(".jpg", resized, encode_param)
        if success and len(encoded) <= max_kb * 1024:
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(encoded.tobytes())
            return encoded.tobytes()
        quality -= 5
        encode_param[1] = quality
    return encoded.tobytes()

def send_csv_over_lora(timestamp, baby_status, cry_status, cropped_img_name):
    msg = f"csv,{timestamp},{baby_status},{cry_status},{cropped_img_name}?"
    lora.write((msg + "\n").encode())
    print(f"Sent CSV over LoRa: {msg}")
    time.sleep(0.5)

def send_image_over_lora(image_path):
    if not os.path.exists(image_path):
        print("Image not found.")
        return

    image_name = os.path.basename(image_path)
    print(f"Sending image: {image_name}")

    img = cv2.imread(image_path)
    compressed = compress_to_96x96_under_2kb(img)
    base64_str = base64.b64encode(compressed).decode("utf-8")

    chunks = [base64_str[i:i+chunk_size] for i in range(0, len(base64_str), chunk_size)]
    total_chunks = len(chunks)

    lora.write(f"IMGSTART:{total_chunks}\n".encode())
    time.sleep(0.5)

    for i, chunk in enumerate(chunks):
        lora.write((chunk + "\n").encode())
        print(f"Chunk {i+1}/{total_chunks}")
        time.sleep(0.5)

    lora.write("IMGEND\n".encode())
    print("Image transmission complete.\n")
    time.sleep(2)

# === Audio Callback ===
def audio_callback(result: audio.AudioClassifierResult, timestamp_ms: int):
    global cry_detected_flag
    for category in result.classifications[0].categories:
        if "cry" in category.category_name.lower() and category.score > score_threshold:
            cry_detected_flag = True

# === MediaPipe Audio Setup ===
base_options = python.BaseOptions(model_asset_path=audio_model_path)
audio_options = audio.AudioClassifierOptions(
    base_options=base_options,
    running_mode=audio.RunningMode.AUDIO_STREAM,
    max_results=max_results,
    score_threshold=score_threshold,
    result_callback=audio_callback
)
classifier = audio.AudioClassifier.create_from_options(audio_options)

buffer_size = 15600
sample_rate = 16000
num_channels = 1
audio_format = containers.AudioDataFormat(num_channels, sample_rate)
record = audio_record.AudioRecord(num_channels, sample_rate, buffer_size)
audio_data = containers.AudioData(buffer_size, audio_format)

def detect_cry_from_mic_stream(timeout_sec=5):
    global cry_detected_flag, global_timestamp_ms
    cry_detected_flag = False
    print("Audio check started...")

    record.start_recording()
    buffer_duration_ms = int((buffer_size / sample_rate) * 1000)
    timestamp_ms = global_timestamp_ms
    start = time.time()

    while time.time() - start < timeout_sec:
        data = record.read(buffer_size)
        audio_data.load_from_array(data)
        classifier.classify_async(audio_data, timestamp_ms)
        timestamp_ms += buffer_duration_ms
        time.sleep(buffer_duration_ms / 1000)
        if cry_detected_flag:
            break

    global_timestamp_ms = timestamp_ms
    return cry_detected_flag

# === MAIN LOOP ===
def main():
    print("Baby Monitoring System Started")

    yolo_model = YOLO(model_path, task="detect")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2.start()
    time.sleep(2)

    last_sent_time = time.time()

    try:
        while True:
            timestamp = get_timestamp()
            postfix = timestamp

            frame = picam2.capture_array()
            if frame is None or frame.size == 0:
                print("⚠️ Skipping invalid frame.")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Save original image
            orig_name = f"original_{postfix}.jpg"
            orig_path = os.path.join(original_dir, orig_name)
            save_compressed_image(frame, orig_path)

            results = yolo_model(frame, imgsz=256)
            best_box = None
            best_score = 0
            for result in results:
                for i, box in enumerate(result.boxes.xyxy.numpy()):
                    score = result.boxes.conf.numpy()[i]
                    if score > best_score:
                        best_box = box
                        best_score = score

            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box)
                cropped = frame[y1:y2, x1:x2]
                cropped_img_name = f"baby_{postfix}.jpg"
                cropped_path = os.path.join(cropped_dir, cropped_img_name)
                reduced_path = os.path.join(reduced_dir, cropped_img_name)

                save_compressed_image(cropped, cropped_path)
                compress_to_96x96_under_2kb(cropped, save_path=reduced_path)

                baby_status = "Detected"
                is_crying = detect_cry_from_mic_stream(timeout_sec=5)
                cry_status = "Crying" if is_crying else "Not Crying"
                print("Baby is crying" if is_crying else "Baby is not crying")

                with open(log_file, "a", newline="") as f:
                    csv.writer(f).writerow([timestamp, baby_status, cry_status, cropped_img_name])

                send_csv_over_lora(timestamp, baby_status, cry_status, cropped_img_name)
                send_image_over_lora(cropped_path)
                last_sent_time = time.time()
            else:
                print("No baby detected.")
                if time.time() - last_sent_time >= HEARTBEAT_INTERVAL:
                    lora.write(b"status:alive\n")
                    print("Heartbeat sent.")
                    last_sent_time = time.time()
                time.sleep(3)

    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
