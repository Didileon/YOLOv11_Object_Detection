from dotenv import load_dotenv
import os
from roboflow import Roboflow

load_dotenv()  # загружаем переменные из .env

api_key = os.getenv("ROBOFLOW_API_KEY")  # получаем ключ

rf = Roboflow(api_key=api_key)  # передаём ключ в Roboflow
workspace = rf.workspace("didi-0uoii")
project = workspace.project("yolov11_object_detection")
version = project.version(1)

dataset = version.download("yolov11")
print(f"Датасет скачан в: {dataset.location}")