import cv2
from ultralytics import YOLO
import os

from ultralytics import YOLO

# Load YOUR best model
model = YOLO('runs/detect/basketball_run1/weights/best.pt')

# Run on the quarter video
model.predict(source="video.mp4", save=True, conf=0.5)

# =================CONFIGURATION=================
# 1. Path to your trained weights
model_path = '/ari/users/aozturk03/YoloV8FT/runs/detect/basketball_7class_run/weights/best.pt'

# 2. Path to your input video
input_video_path = '/ari/users/aozturk03/Basketball_raw_images/sample_game.mp4'

# 3. Define which classes you want to see.
# Based on your training logs: 0=player, 1=referee.
# We only keep these two for a clean look.
CLASSES_TO_SHOW = [0, 1] 

# 4. Output settings
project_dir = '/ari/users/aozturk03/YoloV8FT/inference_output'
run_name = 'clean_demo'
# ===============================================

def run_clean_inference():
    # Ensure output directory exists to avoid errors
    os.makedirs(project_dir, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Processing video: {input_video_path}...")
    print(f"Filtering for class indices: {CLASSES_TO_SHOW}")

    # --- THE MAIN CHANGE ---
    # We use model.predict() instead of model.track().
    # This turns off the ID generation, removing the messy numbers.
    results = model.predict(
        source=input_video_path,
        save=True,              # Save the annotated video
        conf=0.5,               # Only show detections with >50% confidence
        iou=0.45,               # Standard NMS threshold
        classes=CLASSES_TO_SHOW, # <-- FILTER: Only show players and refs
        project=project_dir,
        name=run_name,
        device=0                # Ensure it runs on the GPU
    )

    print("--------------------------------------------------------")
    # The results object is a list (one item per frame for videos),
    # so we check the save_dir of the first result.
    output_folder = results[0].save_dir
    print(f"Clean video saved to: {output_folder}")
    print("You can now download this folder to view the result.")
    print("--------------------------------------------------------")

if __name__ == "__main__":
    run_clean_inference()