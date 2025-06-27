import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved_count:04d}.jpg", frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Извлечено {saved_count} кадров")

# Пример использования
extract_frames("input_video.mp4", "dataset/images")