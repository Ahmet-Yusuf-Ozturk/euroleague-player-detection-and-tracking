from ultralytics import YOLO
import wandb

# 1. Init W&B
wandb.init(project="basketball-cv-final", name="yolov8m-7class-ft")

# 2. Load Model
model = YOLO('yolov8m.pt')  # Using Medium model for balance

# 3. Train
results = model.train(
    data=DATA_DIR,
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.01,
    name='basketball_7class_run',
    device=0
)

wandb.finish()