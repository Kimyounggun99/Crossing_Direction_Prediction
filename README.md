# Pedestrian Crossing Intention Prediction at Intersections

This repository contains the implementation for pedestrian crossing intention prediction using a combination of RTMPose3D, PedCMT, and PedGraph+ frameworks. The project involves data preparation, feature extraction, and training/testing a Transformer-based model.

---

## 1. Environment
To ensure compatibility and smooth execution, please provide the following details of your system environment:
- Operating System (e.g., Ubuntu 20.04, Windows 10)
- Python version (e.g., Python 3.8)
- GPU and CUDA version (if applicable)
- Required Python libraries (e.g., PyTorch, OpenMMLab packages)

---

## 2. Requirements
This project builds upon the following repositories. Please clone them and refer to their documentation:
- **RTMPose3D**: [GitHub Repository](https://github.com/open-mmlab/RTMPose3D)
- **PedCMT**: [GitHub Repository](https://github.com/your-org/PedCMT)
- **PedGraph+**: [GitHub Repository](https://github.com/your-org/PedGraphPlus)

Ensure you have the necessary dependencies installed for each repository before proceeding.

---

## 3. Data Preparation and Feature Extraction

### Data Preparation
Prepare your dataset by organizing videos from the same CCTV source in the same folder. Define the **target waiting area** for pedestrians and the **crosswalk direction** reference point `{I}`.

### Feature Extraction

#### Example Image:
*(Attach a visual representation of the target area and pedestrian detection zones.)*

#### Steps:
1. Extract the target area using the following command:
   ```bash
   python Extract_target_area --data_path {your video path}
