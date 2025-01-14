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
   ```
2. Assign the pixel values of your target waiting area to the 'Target_regio'n variable in 'Feature_extraction'.py.

3. Map the pixel values for '{I}' to 'p1', 'p2', and 'p3' in 'mapping_2d_to_3d.py'.

4. Extract pedestrian features and save them as a CSV file using:
   ```bash
   python Features_extraction.py configs/rtmdet_m_640-8xb32_coco-person.py \
   https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
   configs/rtmw3d-l_8xb64_cocktail14-384x288.py \
   https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth \
   --input '{./Your_path/cam_num/input_video.avi}' \
   --cam_num {specify camera number based on the Target_region} \
   --root_path '{./Your_path/cam_num}' \
   --output-root '{./Your_path/output}'
   ```

## 4. Model Training & Testing

### Training
Train the model using the following command:

   ```bash
   python main.py --observation-time {Specify observation time} \
   --experiment-type {TTE(Time to Event)/SEQ(All waiting period)} \
   --mode train \
   --model Transformer_based_model \
   --save_dir {./Your_path/checkpoints} \
   --output_dir {./Your_path/logs}
   ```

### Testing
Test the model using:
   ```bash
   python main.py --observation-time {Specify observation time} \
   --experiment-type {TTE(Time to Event)/SEQ(All waiting period)} \
   --mode test \
   --model Transformer_based_model \
   --load {./Your_path/checkpoints/checkpoint_file.pth}
   ```
