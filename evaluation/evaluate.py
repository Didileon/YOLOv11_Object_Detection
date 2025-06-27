from ultralytics import YOLO
import pandas as pd

model = YOLO("runs/detect/train/weights/best.pt")  # создаём модель один раз

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

if __name__ == "__main__":
    evaluate_model()
    predict_on_video()