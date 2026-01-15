# EuroLeague Player Tracking

## Overview

This repository contains an end-to-end computer vision pipeline for **real-time detection and tracking of players and key game objects in EuroLeague basketball games**. The project is designed as a foundational step toward **automated sports narration**, with a particular focus on improving accessibility for visually impaired fans.

Unlike NBA-focused solutions, EuroLeague games present unique visual challenges such as **smaller courts, higher player density, frequent occlusions, and diverse arena designs**. This project addresses those challenges through domain-specific data collection, auto-labeling, and model fine-tuning.

---

## Motivation

The primary motivation behind this project is **social accessibility**. Inspired by manual narration efforts for visually impaired fans, the long-term goal is to automate the perception layer of basketball games so that detected events can later be translated into natural language commentary.

This repository focuses on the **vision component** of that pipeline: enabling a model to reliably “see” the game in real time.

---

## Pipeline Summary

The project consists of three main components:

1. **Dataset Creation**

   * Extracted frames from EuroLeague match footage
   * Sampling interval: 0.5 seconds
   * Total dataset size: 672 images

2. **Auto-Labeling**

   * Initial experiments with Grounding DINO revealed significant misclassifications in crowded scenes
   * Switched to **SAM3**, which produced substantially higher-quality annotations
   * Custom class design to reduce false positives

3. **Model Fine-Tuning**

   * Fine-tuned a **YOLOv8m** model on the auto-labeled dataset
   * Optimized for low-latency, real-time inference
   * Evaluated using training metrics and qualitative video-based testing

---

## Detection Classes

The model is trained to detect the following classes:

* Player
* Jersey Number
* Referee
* Ball
* Basketball Rim
* Ad Basketball (introduced to avoid confusion with real game ball)
* Shot Clock

These classes were chosen to support future extensions such as player identification, team clustering, and event logging.

---

## Results

### Quantitative Evaluation

* Training and validation loss curves indicate successful convergence
* Mean Average Precision (mAP) metrics were tracked during training

*(Note: Detailed numerical results can be added here as the project evolves.)*

### Qualitative Evaluation

* The fine-tuned model successfully detects players and key objects in **real-time game footage**
* Demonstrated robustness under occlusions and dense player interactions
* In-game detection examples and demo videos are provided via links in the project references

---

## Demo

Sample qualitative results are provided in the repository:

- `demo_images/`: Example frames with detection outputs
- `demo_videos/`: Inference results on full game clips

These demonstrations showcase real-time detection performance under dense player interactions and occlusions.

---

## Limitations

* Jersey number detection is currently unreliable when trained directly on full-frame images
* Player re-identification is limited when players leave and re-enter the frame
* Current demonstrations focus on a limited number of games

These limitations are explicitly acknowledged and addressed in the future work roadmap.

---

## Future Work

Planned extensions include:

* **Two-stage jersey number recognition** (player crop + OCR/classifier)
* **Player re-identification** using jersey numbers and learned embeddings
* **Team clustering** to differentiate players with identical jersey numbers
* **Homography-based court mapping** for top-down player localization
* **Event detection and logging** to enable automated narration via generative models

---

## Tech Stack

* Python
* YOLOv8 (Ultralytics)
* SAM3
* OpenCV
* PyTorch

---

## Project Status

This project was developed as part of a graduate-level course (DSAI 544) and is actively evolving. Additional experiments on different games and environments are ongoing.

---

## Acknowledgements

* Ultralytics YOLOv8
* SAM3
* Grounding DINO
* Inspiration from Basketball AI projects and accessibility-focused sports initiatives

---

## License

This project is provided for educational and research purposes. Please check individual dependencies for their respective licenses.
