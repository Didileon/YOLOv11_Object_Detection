from roboflow import Roboflow
import shutil

def prepare_dataset(api_key, workspace, project_name):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(1).download("yolov11")
    
    # Автоматическое разделение train/val/test
    dataset.split(test=0.1, val=0.2)
    
    # Аугментация (пример)
    dataset.augment(
        rotation=(-15, 15),
        brightness=(0.8, 1.2),
        flip_horizontal=True
    )
    shutil.move(dataset.location, "final_dataset")

prepare_dataset("ВАШ_API_KEY", "WORKSPACE_NAME", "PROJECT_NAME")