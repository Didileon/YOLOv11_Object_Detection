
import os

print("Текущая рабочая директория:", os.getcwd())

import cv2

def extract_frames(video_path, output_dir, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видеофайл {video_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved_count:04d}.jpg", frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Из видео {os.path.basename(video_path)} извлечено {saved_count} кадров")

def process_all_videos(raw_dir, processed_dir, frame_interval=10):
    for filename in os.listdir(raw_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(raw_dir, filename)
            # Создаём отдельную папку с именем видео без расширения
            video_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(processed_dir, video_name)
            extract_frames(video_path, output_dir, frame_interval)

if __name__ == "__main__":
    raw_videos_folder = "dataset/raw"
    processed_images_folder = "dataset/processed/images"
    process_all_videos(raw_videos_folder, processed_images_folder, frame_interval=10)