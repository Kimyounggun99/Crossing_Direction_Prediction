# Pedestrian Crossing Intention Prediction at Intersections


![GIF Preview](Geo_invariant_gif.gif)

## 1. Environment
To ensure compatibility and smooth execution, please provide the following details of your system environment:
- Operating System: Ubuntu 22.04
- Python version: Python 3.8
- CUDA version: 12.4
---

## 2. Requirements
This project builds upon [RTMPose3D](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), [PedCMT](https://github.com/xbchen82/PedCMT), [PedGraph+](https://github.com/RodrigoGantier/Pedestrian_graph_plus.git) repositories. Please refer to their documentation for requirements. Ensure you have the necessary dependencies installed for each repository before proceeding.

---

## 3. Data Preparation and Feature Extraction

### 3-1. Data Preparation for own data usage
If you want to use our key points and trajectories data, you can skip this step.

Prepare your dataset by organizing videos from the same CCTV source in the same folder. Define the **target waiting area** for pedestrians and the **crosswalk direction** reference point `{I}`.

#### Feature Extraction
![intersection](https://github.com/user-attachments/assets/7af85e64-2b0e-4003-9620-f53f6972462a)


#### Steps:
1. Extract the target area using the following command:
   ```bash
   python Extract_target_area --data_path {your video path}
   ```
2. Assign the pixel values of your target waiting area to the `Target_region` variable in `Feature_extraction`.py.

3. Map the pixel values for `{I}` to `p1`, `p2`, and `p3` in `utils.py`.

4. Extract pedestrian features and save them as a CSV file using:
   ```bash
   python Features_extraction.py configs/rtmdet_m_640-8xb32_coco-person.py \
   https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
   configs/rtmw3d-l_8xb64_cocktail14-384x288.py \
   https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth \
   --input {./Your_path/cam_num/input_video.avi} \
   --cam_num {specify camera number based on the Target_region} \
   --root_path {./Your_path/cam_num} \
   --output-root {./Your_path/output}
   ```

### 3-2. Data Preparation to use our data.
Since visual information from CCTV may occur important privacy issues, We provide only trajectories and key points.
Download all csv files from [OneDrive](https://ucf-my.sharepoint.com/my?id=%2Fpersonal%2Fyo171134%5Fucf%5Fedu%2FDocuments%2FCrossing%20Dirrection%20Prediction%2FCCTV%5Fdata&login_hint=yo171134%40ucf%2Eedu) and build following data structure:

```bash
./YourPath/CCTV_data
├── train
├── test
```

## 4. Model Zoo: TODO

We provide each model's weight trained on the observation time window 4 seconds.

| Model | Accuracy | F1-Score | Precision | Recall | Download |
|----------|----------------|---------------|----------------------|-----------------|--------------------|
| GCN |  |  |  |  | [GCN](https://ucf-my.sharepoint.com/my?id=%2Fpersonal%2Fyo171134%5Fucf%5Fedu%2FDocuments%2FCrossing%20Dirrection%20Prediction%2FGCN&login_hint=yo171134%40ucf%2Eedu) |
| Transformer |  |  |  |  | [Transformer](https://ucf-my.sharepoint.com/my?id=%2Fpersonal%2Fyo171134%5Fucf%5Fedu%2FDocuments%2FCrossing%20Dirrection%20Prediction%2FTransformer&login_hint=yo171134%40ucf%2Eedu) |
| Transformer+GCN |  |  |  |  | [GCN+Transformer](https://ucf-my.sharepoint.com/my?id=%2Fpersonal%2Fyo171134%5Fucf%5Fedu%2FDocuments%2FCrossing%20Dirrection%20Prediction%2FTransformer%2BGCN&login_hint=yo171134%40ucf%2Eedu) |

## 5. Model Training 
#### You can skip this step if you use our weight files.
Train the model using the following command:

   ```bash
   python main.py --observation-time {Specify observation time(1-4)} \
   --experiment-type {TTE(Time to Event)/SEQ(All waiting period)} \
   --input_dir {./YourPath/CCTV_data} \
   --mode train \
   --model {Transformer_based_model/GCN_based_model/Transformer_GCN_mixing_model} \
   --save_dir {./Your_path/checkpoints} \
   --output_dir {./Your_path/logs}
   ```

## 6. Testing
Test the model using:
   ```bash
   python main.py --observation-time {Specify observation time} \
   --experiment-type {TTE(Time to Event)/SEQ(All waiting period)} \
   --input_dir {./YourPath/CCTV_data} \
   --mode test \
   --model {Transformer_based_model/GCN_based_model/Transformer_GCN_mixing_model} \
   --load {./Your_path/checkpoints/checkpoint_file.pth}
   ```


## Acknowledgment
This repository is based on [RTMPose3D](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), [PedCMT](https://github.com/xbchen82/PedCMT), [PedGraph+](https://github.com/RodrigoGantier/Pedestrian_graph_plus.git) repositories. 

