from ultralytics import YOLO
import pandas as pd
import os
import matplotlib.pyplot as plt

# 1. Создаём модель
model = YOLO("C:/Users/User/Desktop/PycharmProjects/YOLOv11_Object_Detection/YOLOv11_Object_Detection-1/yolo11s.pt")

def evaluate_model():
    metrics = model.val(
        data="final_dataset/data.yaml",
        split="test"
    )
    # Экспорт метрик
    report = {
        "mAP@0.5": metrics.box.map50,
        "Precision": metrics.box.p,
        "Recall": metrics.box.r,
        "F1": metrics.box.f1
    }
    pd.DataFrame([report]).to_csv("final_metrics.csv")

def predict_on_video():
    model.predict(
        source="test_video.mp4",
        conf=0.5,
        save=True,
        project="results"
    )

def plot_training_curves(results_csv_path="runs/detect/train/results.csv"):
    if not os.path.exists(results_csv_path):
        print(f"Файл {results_csv_path} не найден.")
        return

    df = pd.read_csv(results_csv_path)
    # График потерь (loss)
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')
    plt.legend()
    plt.title('Box Loss per Epoch')
    plt.savefig("loss_manual.png")
    plt.close()

    # График метрик (например, mAP@0.5)
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.title('mAP@0.5 per Epoch')
    plt.savefig("metrics_manual.png")
    plt.close()

    print("Графики сохранены: loss_manual.png, metrics_manual.png")

if __name__ == "__main__":
    evaluate_model()
    predict_on_video()
    plot_training_curves("runs/detect/train/results.csv")

# Проверка наличия файла весов
print(os.path.exists("C:/Users/User/Desktop/PycharmProjects/YOLOv11_Object_Detection/YOLOv11_Object_Detection-1/yolo11s.pt"))