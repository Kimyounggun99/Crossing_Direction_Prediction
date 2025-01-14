# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from typing import List

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log
import copy
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.utils import adapt_mmdet_pipeline

from mmpose.visualization.local_visualizer_3d import Pose3dLocalVisualizer

import supervision as sv
import time

from scipy.optimize import linear_sum_assignment

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from rtmpose3d import *  

from collections import defaultdict, deque

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from utils import transformation, Mapping_real, get_img_coord, scale_z_values
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable

import glob
import os
import pandas as pd
import gc

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose3d_estimator_config',
        type=str,
        default=None,
        help='Config file for the 3D pose estimator')
    parser.add_argument(
        'pose3d_estimator_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for the 3D pose estimator')
    parser.add_argument('--input', type=str, default='', help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to show visualizations')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        default=False,
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='Whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.5,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='Inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the 2D pose'
        'detection stage. Default: False.')
    parser.add_argument(
        '--dim',
        type=int,
        default=2,
        help="2D/3D pose estimation"
    )
    parser.add_argument(
        '--confidence_threshold',
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    parser.add_argument('--cam_num',type=str,help='Camera number')
    parser.add_argument('--root_path', type=str)
    args = parser.parse_args()
    return args




def assign_ids(pose_est_results, bbox_id, iou_threshold=0.3, next_id=0):

    def compute_iou(box1, box2):
        """Compute the IoU between two bounding boxes."""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    for pose_est_result in pose_est_results:
        bbox = pose_est_result.pred_instances.bboxes[0]  # Assuming one bbox per result
        best_iou = 0
        best_id = None

        # Compare with existing bbox_id
        for existing_id, existing_bbox in bbox_id.items():
            iou = compute_iou(bbox, existing_bbox[0])
            if iou > best_iou:
                best_iou = iou
                best_id = existing_id

        # Assign ID based on IoU
        if best_iou >= iou_threshold:
            pose_est_result.track_id = best_id
            bbox_id[best_id]= bbox.reshape(1, -1)
        else:
            # Create a new ID if no sufficient overlap
           
            pose_est_result.track_id = next_id
            bbox_id[next_id] = bbox.reshape(1, -1)
            next_id += 1

    return pose_est_results, next_id, bbox_id



def process_one_image(args, detector, frame: np.ndarray, frame_idx: int,
                      pose_estimator,
                      pose_est_results_last: List[PoseDataSample],
                      pose_est_results_list: List[List[PoseDataSample]],
                      next_id: int, visualize_frame: np.ndarray,
                      visualizer: Pose3dLocalVisualizer, bbox_id= None, key_point_id= None):
    """Visualize detected and predicted keypoints of one image.

    Pipeline of this function:

                              frame
                                |
                                V
                        +-----------------+
                        |     detector    |
                        +-----------------+
                                |  det_result
                                V
                        +-----------------+
                        |  pose_estimator |
                        +-----------------+
                                |  pose_est_results
                                V
                       +-----------------+
                       | post-processing |
                       +-----------------+
                                |  pred_3d_data_samples
                                V
                         +------------+
                         | visualizer |
                         +------------+

    Args:
        args (Argument): Custom command-line arguments.
        detector (mmdet.BaseDetector): The mmdet detector.
        frame (np.ndarray): The image frame read from input image or video.
        frame_idx (int): The index of current frame.
        pose_estimator (TopdownPoseEstimator): The pose estimator for 2d pose.
        pose_est_results_last (list(PoseDataSample)): The results of pose
            estimation from the last frame for tracking instances.
        pose_est_results_list (list(list(PoseDataSample))): The list of all
            pose estimation results converted by
            ``convert_keypoint_definition`` from previous frames. In
            pose-lifting stage it is used to obtain the 2d estimation sequence.
        next_id (int): The next track id to be used.
        pose_lifter (PoseLifter): The pose-lifter for estimating 3d pose.
        visualize_frame (np.ndarray): The image for drawing the results on.
        visualizer (Visualizer): The visualizer for visualizing the 2d and 3d
            pose estimation results.

    Returns:
        pose_est_results (list(PoseDataSample)): The pose estimation result of
            the current frame.
        pose_est_results_list (list(list(PoseDataSample))): The list of all
            converted pose estimation results until the current frame.
        pred_3d_instances (InstanceData): The result of pose-lifting.
            Specifically, the predicted keypoints and scores are saved at
            ``pred_3d_instances.keypoints`` and
            ``pred_3d_instances.keypoint_scores``.
        next_id (int): The next track id to be used.
    """
    # pose_dataset = pose_estimator.cfg.test_dataloader.dataset
    pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']
    
    # First stage: conduct 2D pose detection in a Topdown manner
    # use detector to obtain person bounding boxes
    det_result = inference_detector(detector, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()
    
    # filter out the person instances with category and bbox threshold
    # e.g. 0 for person in COCO
    bboxes = pred_instance.bboxes
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    # estimate pose results for current image
 
    pose_est_results = inference_topdown(pose_estimator, frame, bboxes)
    
    if visualizer is None:
        if len(pose_est_results_last)>0:  
            pose_est_results, next_id, bbox_id= assign_ids(pose_est_results, bbox_id, iou_threshold=0.2, next_id=next_id)

    # post-processing
    for idx, pose_est_result in enumerate(pose_est_results):
        if visualizer is None:
            if 'track_id' not in pose_est_result:
                pose_est_result.track_id = next_id
                bbox_id[next_id]= pose_est_result.pred_instances.bboxes
                key_point_id[next_id]= pose_est_result.pred_instances.transformed_keypoints

                next_id += 1
        else:
            pose_est_result.track_id = pose_est_results[idx].get('track_id', 10)
        
        pred_instances = pose_est_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_est_results[
                idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)
        
        keypoints = -keypoints[..., [0, 2, 1]]

        # rebase height (z-axis)
        if not args.disable_rebase_keypoint:
            keypoints[..., 2] -= np.min(
                keypoints[..., 2], axis=-1, keepdims=True)

        pose_est_results[idx].pred_instances.keypoints = keypoints
    
    pose_est_results = sorted(
        #pose_est_results, key=lambda x: x.get('track_id'))
        pose_est_results, key=lambda x: x.get('track_id', 1e4))
    
    pred_3d_data_samples = merge_data_samples(pose_est_results)
    pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

    if args.num_instances < 0:
        args.num_instances = len(pose_est_results)
    
    if visualizer is not None:
        visualizer.add_datasample(
            'result', #ok
            visualize_frame,
            data_sample=pred_3d_data_samples,
            det_data_sample=pred_3d_data_samples,
            draw_gt=False,
            draw_2d=True,
            dataset_2d=pose_det_dataset_name,
            dataset_3d=pose_det_dataset_name,
            show=args.show,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            convert_keypoint=False,
            axis_limit=400,  #ok
            axis_azimuth=70,
            axis_elev=15,
            num_instances=args.num_instances,
            wait_time=args.show_interval)    
        return pose_est_results, pose_est_results_list, pred_3d_instances, next_id
    else:
        return pred_3d_data_samples, pose_est_results, pose_est_results_list, pred_3d_instances, next_id, bbox_id, key_point_id






def is_point_in_polygon(point, polygon):
    """Check if a point is inside the polygon ROI."""
    return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0

def draw_keypoints_on_frame(frame, keypoints, color=(0, 255, 0)):
    """
    이미지에 keypoints를 그리는 함수
    frame: 이미지 프레임
    keypoints: (N, 2) 크기의 키포인트 좌표 배열
    color: 키포인트의 색상 (BGR)
    """
    keypoints= keypoints[:23] #17        # Index기준 #17:(왼발 엄지), #18(왼발 새끼) 19(왼발 뒷꿈치),    20(오른발 엄지) 21(오른 새끼) 22(오른발 뒷꿈치)
    for point in keypoints:
        x, y = point
        # 각 keypoint 위치에 원을 그립니다.
        cv2.circle(frame, (int(x), int(y)), radius=3, color=color, thickness=-1)
    return frame



def draw_skeleton_on_frame(frame, keypoints, ids, color=(255, 0, 0)):
    skeleton = [
    (0, 1), (0, 2), 
    (0, 5), (0, 6),# 머리와 어깨
    (5, 7), (7, 9),          # 상체
    (5, 6), (5, 11),  (8,6), (8,10) ,       # 팔
    (11, 12), (12, 6),          # 몸통과 엉덩이
    (11, 13), (13, 15), (15,19), (19,17), (19,18),       # 왼다리
    (12, 14), (14, 16), (16,20), (20,21), (20,22),      # 오른다리
    
    ]   
    colors = [
    (255, 0, 0),   # 파란색
    (0, 255, 0),   # 초록색
    (0, 0, 255),   # 빨간색
    (255, 255, 0), # 하늘색
    (255, 0, 255), # 보라색
    (0, 255, 255), # 노란색
    (128, 0, 128), # 보라색
    (0, 128, 255), # 주황색
    (128, 128, 0), # 올리브색
    (0, 128, 255), # 주황색
    (128, 128, 0), # 올리브색
    (255, 128, 0), # 밝은 주황색
    (255, 0, 128), # 핑크색
    (128, 0, 255), # 보라색
    (0, 255, 128), # 청록색
    (128, 128, 0), # 올리브색
    (0, 128, 255), # 주황색
    (128, 128, 0), # 올리브색
    (255, 128, 0), # 밝은 주황색
    (255, 0, 128), # 핑크색
    (128, 0, 255), # 보라색
    (0, 255, 128), # 청록색
    ]
    for i, joint in enumerate(skeleton):
        
        pt1 = tuple(keypoints[joint[0]].astype(int))
        pt2 = tuple(keypoints[joint[1]].astype(int))
        color = colors[i % len(colors)]
        cv2.line(frame, pt1, pt2, color, 2)

    head_keypoint = tuple(keypoints[0].astype(int))
                        # Ensure the text position is within the image boundaries
    text_position = (
        max(0, min(frame.shape[1] - 1, head_keypoint[0])),
        max(0, min(frame.shape[0] - 1, head_keypoint[1] - 10))
    )

    cv2.putText(frame, f'ID: {ids}', text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    return frame




def plot_3d_keypoints_from_position_norm(positions,scores=None, ax=None):
    """
    3D keypoints와 링크를 시각화.
    
    Args:
        positions (dict): keypoints의 3D 좌표.
        links (list): 연결된 keypoints 쌍의 리스트. 예: [('L_ankle', 'L_knee'), ('L_knee', 'L_pelvis')].
        ax (Axes3D): 기존 3D 축. 없으면 새로 생성.
    """





    links=[
                ('L_ankle', 'L_knee'), ('L_knee', 'L_pelvis'),
                ('L_pelvis', 'R_pelvis'), ('R_pelvis', 'R_knee'),
                ('R_knee', 'R_ankle'), ('L_pelvis', 'L_shoulder'),
                ('R_pelvis', 'R_shoulder'), ('L_shoulder', 'R_shoulder'),
                ('L_shoulder', 'L_elbow'), ('L_elbow', 'L_wrist'),
                ('R_shoulder', 'R_elbow'), ('R_elbow', 'R_wrist'),
                ('L_shoulder', 'M_shoulder'),('M_shoulder', 'head'), 
    ]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    x = [0, 2]  
    y = [0, 0]  
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [1, 2]  
    y = [1, 1]  
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [1.25, 1.25]  
    y = [0, 1]  
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [1.5, 1.5]  
    y = [0, 1]  
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [1.75, 1.75] 
    y = [0, 1] 
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")

    
    ax.text(1, 0, 0, 'A', color='red', fontsize=12, fontweight='bold')

    
    x = [0, 0] 
    y = [2, 0] 
    z = [0, 0] 
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (0,1,0)")
    
    
    x = [1, 1] 
    y = [1, 2] 
    z = [0, 0] 
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [0, 1]  
    y = [1.25, 1.25] 
    z = [0, 0] 
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [0, 1] 
    y = [1.5, 1.5]  
    z = [0, 0]  
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [0, 1] 
    y = [1.75, 1.75]  
    z = [0, 0]  
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    ax.text(0, 1, 0, 'B', color='blue', fontsize=12, fontweight='bold')


    square_x = [0, 1, 1, 0, 0]
    square_y = [0, 0, 1, 1, 0]
    square_z = [0, 0, 0, 0, 0]
    ax.plot(square_x, square_y, square_z, color='black', linewidth=2)


    cmap = get_cmap('viridis') 
    norm = Normalize(vmin=0, vmax=1)  

    for key, pos in positions.items():
        color = 'blue' 
        if scores and key in scores:
            color = cmap(norm(scores[key])) 
        ax.scatter(*pos, label=key, s=15, color=color)
        if key in ['L_ankle', 'R_ankle', 'head']:
            ax.text(*pos, key, fontsize=8)

    for link in links:
        if link[0] in positions and link[1] in positions:
            point1 = positions[link[0]]
            point2 = positions[link[1]]
            ax.plot(
                [point1[0], point2[0]],
                [point1[1], point2[1]],
                [point1[2], point2[2]],
                'b'
            )


    if 'head' in positions and 'M_shoulder' in positions:
        head = positions['head']
        shoulder = positions['M_shoulder']
        head_dir = head - shoulder  # 방향 벡터

        # Z축 성분 제거 (XY 평면 투영)
        head_dir[2] = 0  

        # 벡터 스케일링 (크기를 2로 만들기 위해 정규화 후 곱하기)
        magnitude = np.linalg.norm(head_dir[:2])  # XY 평면에서의 크기만 고려
        if magnitude != 0:  # 벡터가 영벡터가 아닐 때만 정규화
            head_dir = (head_dir / magnitude) * 0.2
        else:
            head_dir = np.zeros_like(head_dir)

        

        # 시각화: XY 평면 방향 벡터
        ax.quiver(
            head[0], head[1], head[2],  # 시작점 (head 위치)
            head_dir[0], head_dir[1], 0,  # XY 평면 방향 벡터 (Z축 = 0)
            color='red', linewidth=2, label='Head Direction'
        )

    # 축 범위 설정
    all_points = np.array(list(positions.values()))
    #min_range = np.min(all_points, axis=0)-20
    #max_range = np.max(all_points, axis=0)+20
    ax.set_xlim([-0.1,1.1]) #([min_range[0],max_range[0]])
    ax.set_ylim([-0.1,1.1]) #([min_range[1],max_range[1]])
    ax.set_zlim([0,0.8])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=31, azim=18)  # 초기 뷰 설정

    # 컬러바 추가
    #sm = ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])  # ScalarMappable에 빈 데이터 설정 (컬러바만 사용)
    #cbar = plt.colorbar(sm, ax=ax,  shrink=0.4)
    #cbar.set_label('Confidence Score')


def plot_3d_keypoints_from_position_only_axis(ax=None):
    """
    3D keypoints와 링크를 시각화.
    
    Args:
        positions (dict): keypoints의 3D 좌표.
        links (list): 연결된 keypoints 쌍의 리스트. 예: [('L_ankle', 'L_knee'), ('L_knee', 'L_pelvis')].
        ax (Axes3D): 기존 3D 축. 없으면 새로 생성.
    """



    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


    x = [0, 2]  
    y = [0, 0]  
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [1, 2]  
    y = [1, 1]  
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [1.25, 1.25]  
    y = [0, 1]  
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [1.5, 1.5]  
    y = [0, 1]  
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [1.75, 1.75]  
    y = [0, 1]  
    z = [0, 0]  
    ax.plot(x, y, z, color='red', linewidth=2, label="Line (0,0,0) to (1,0,0)")

    
    ax.text(1, 0, 0, 'A', color='red', fontsize=12, fontweight='bold')

    
    x = [0, 0]  # x 좌표
    y = [2, 0]  # y 좌표
    z = [0, 0]  # z 좌표
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (0,1,0)")
    
    
    x = [1, 1]  # x 좌표
    y = [1, 2]  # y 좌표
    z = [0, 0]  # z 좌표
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [0, 1]  # x 좌표
    y = [1.25, 1.25]  # y 좌표
    z = [0, 0]  # z 좌표
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [0, 1]  # x 좌표
    y = [1.5, 1.5]  # y 좌표
    z = [0, 0]  # z 좌표
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    x = [0, 1]  # x 좌표
    y = [1.75, 1.75]  # y 좌표
    z = [0, 0]  # z 좌표
    ax.plot(x, y, z, color='blue', linewidth=2, label="Line (0,0,0) to (1,0,0)")
    ax.text(0, 1, 0, 'B', color='blue', fontsize=12, fontweight='bold')


    # 검정색 네모 그리기
    square_x = [0, 1, 1, 0, 0]
    square_y = [0, 0, 1, 1, 0]
    square_z = [0, 0, 0, 0, 0]
    ax.plot(square_x, square_y, square_z, color='black', linewidth=2)

   
    ax.set_xlim([-0.1,1.1]) #([min_range[0],max_range[0]])
    ax.set_ylim([-0.1,1.1]) #([min_range[1],max_range[1]])
    ax.set_zlim([0,0.8])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=31, azim=18)  # 초기 뷰 설정





def plot_for_multiple_pedestrians(all_positions,all_score=None):
    """
    여러 pedestrian의 3D keypoints를 하나의 3D 플롯에 그립니다.
    
    Args:
        all_positions (list): 각 pedestrian의 keypoints dictionary를 포함한 리스트.
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # 각 pedestrian에 대해 keypoints 시각화
    if len(all_positions):
        for i in range(len(all_positions)):
            positions= all_positions[i]
            scores= all_score[i]
            plot_3d_keypoints_from_position_norm(positions,scores=scores, ax=ax)
    else:
        plot_3d_keypoints_from_position_only_axis(ax=ax)

    # 플롯을 이미지로 변환
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img




def main():
    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_model(
        args.pose3d_estimator_config,
        args.pose3d_estimator_checkpoint,
        device=args.device.lower())

    det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get(
        'skeleton_links', None)
    det_dataset_link_color = pose_estimator.dataset_meta.get(
        'skeleton_link_colors', None)

    pose_estimator.cfg.model.test_cfg.mode = 'vis'
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.line_width = args.thickness
    pose_estimator.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_estimator.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.det_dataset_link_color = det_dataset_link_color  # noqa: E501
    pose_estimator.cfg.visualizer.skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.link_color = det_dataset_link_color
    pose_estimator.cfg.visualizer.kpt_color = det_kpt_color
    
    if args.dim==3:
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    else:
        visualizer= None

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if args.output_root == '':
        save_output = False
    else:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'
        save_output = True

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    pose_est_results_list = []
    pred_instances_list = []
    if input_type == 'image':
        frame = mmcv.imread(args.input, channel_order='rgb')
        _, _, pred_3d_instances, _ = process_one_image(
            args=args,
            detector=detector,
            frame=args.input,
            frame_idx=0,
            pose_estimator=pose_estimator,
            pose_est_results_last=[],
            pose_est_results_list=pose_est_results_list,
            next_id=0,
            visualize_frame=frame,
            visualizer=visualizer)

        if args.save_predictions:
            # save prediction results
            pred_instances_list = split_instances(pred_3d_instances)

        if save_output:
            frame_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(frame_vis), output_file)
    
    elif input_type in ['webcam', 'video']:
        next_id = 0
        
        pose_est_results = []

        if args.input == 'webcam':
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(args.input)

        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)

        video_writer = None
        frame_idx = 0
        row_temp={}
        if args.cam_num=='orange_view1':
            Target_region= np.array([[1907,115], [1233,325],[1204,547], [1864,487], [2530,386], [2790,199]])
        elif args.cam_num=='orange_view2':
            Target_region= np.array([[1507, 524], [962, 1115], [1086, 1433], [1700, 1225], [2550, 407],  [2072, 262]])
        elif args.cam_num=='orange_view3':
            Target_region= np.array([[ 2513, 314], [2490, 444 ], [1227, 323 ], [1319, 227]])
        elif args.cam_num=='orange_view4':
            Target_region= np.array([[674,894],[674,1326],[ 2403, 1216], [2308, 808],[2173,651], [1484,804]])
        elif args.cam_num=='seminole_view1':
            Target_region= np.array([[1501,853 ], [ 1432,1164], [ 2346,1118], [2772,902], [2539,645],[2005,614]])
        elif args.cam_num=='seminole_view2':
            Target_region= np.array([[1994, 522 ], [2386,588], [2496,769], [ 2307,755], [1703, 669 ], [1550, 582],[1498,507], [1530,458], [1723,429]])
        elif args.cam_num=='seminole_view3':
            Target_region= np.array([[1870, 839 ], [1040,1145], [1002,1606], [ 2046, 1450], [2493, 1095 ], [2608, 790], [2510,648], [2276, 602]])
        elif args.cam_num=='seminole_view4':
            Target_region= np.array([[2057,793], [2104,945], [1685, 1015 ], [1365, 914 ], [1201, 741], [1371,559], [1463,568]])
        
        elif args.cam_num=='orange_view1_sideview':
            Target_region= np.array([[2580,1925], [2281,811], [1371,1054], [239,1570]])
        elif args.cam_num=='orange_view2_sideview':
            Target_region= np.array([[2302,1578], [2219,914], [1597,979], [792,1160]])
        else:
            breakpoint()
        
        print(f"INFO:: ################# Camera {args.cam_num} #########################")
        
        folder_path = args.root_path
        print(f"Processing folder: {folder_path}")
        
        # 폴더 내 모든 비디오 파일 (*.mp4, *.avi 등)을 탐색
        video_files = glob.glob(os.path.join(folder_path, "**", "*.*"), recursive=True)
        video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for video_path in video_files:
            
            print(f"Reading video: {video_path}")
            
            mmengine.mkdir_or_exist(video_path.split('\\')[:-1][0]+'Kim')
            video_info = sv.VideoInfo.from_video_path(video_path=video_path)
            output_file= video_path.split('\\')[:-1][0]+'Kim\\'+video_path.split('\\')[-1]
            video_writer = None
            frame_idx = 0
            
            output_csv= output_file.split('.')[:-1][0]+'.csv'
            
            columns = ["frame_id", "ped_id", "key_points_scores", "key_points_positions", "raw_key_points","key_points_position_2ds", "bboxes_2d","bbox_scores_2d"]
            data_2 = []
            row= defaultdict(lambda: {})
            video = cv2.VideoCapture(video_path)
            
            
            
            
            
            key_points_position= {
                                "head":np.zeros([3]), "M_shoulder":np.zeros([3]), "M_pelvis": np.zeros([3]), 
                                "L_shoulder": np.zeros([3]), "L_elbow": np.zeros([3]), "L_wrist": np.zeros([3]),
                                "R_shoulder": np.zeros([3]), "R_elbow": np.zeros([3]), "R_wrist": np.zeros([3]),
                                "L_pelvis": np.zeros([3]), "L_knee": np.zeros([3]), "L_ankle": np.zeros([3]),
                                "R_pelvis": np.zeros([3]), "R_knee": np.zeros([3]), "R_ankle": np.zeros([3]),
                                }
            key_points_position_save= {
                                "head":np.zeros([3]), "M_shoulder":np.zeros([3]), "M_pelvis": np.zeros([3]), 
                                "L_shoulder": np.zeros([3]), "L_elbow": np.zeros([3]), "L_wrist": np.zeros([3]),
                                "R_shoulder": np.zeros([3]), "R_elbow": np.zeros([3]), "R_wrist": np.zeros([3]),
                                "L_pelvis": np.zeros([3]), "L_knee": np.zeros([3]), "L_ankle": np.zeros([3]),
                                "R_pelvis": np.zeros([3]), "R_knee": np.zeros([3]), "R_ankle": np.zeros([3]),
                                }
            key_points_score= {
                                "head":np.zeros([1]), "M_shoulder":np.zeros([1]), "M_pelvis": np.zeros([1]), 
                                "L_shoulder": np.zeros([1]), "L_elbow": np.zeros([1]), "L_wrist": np.zeros([1]),
                                "R_shoulder": np.zeros([1]), "R_elbow": np.zeros([1]), "R_wrist": np.zeros([1]),
                                "L_pelvis": np.zeros([1]), "L_knee": np.zeros([1]), "L_ankle": np.zeros([1]),
                                "R_pelvis": np.zeros([1]), "R_knee": np.zeros([1]), "R_ankle": np.zeros([1]),
                                }
            key_points_position_2d= {
                                "head":np.zeros([2]), "M_shoulder":np.zeros([2]), "M_pelvis": np.zeros([2]), 
                                "L_shoulder": np.zeros([2]), "L_elbow": np.zeros([2]), "L_wrist": np.zeros([2]),
                                "R_shoulder": np.zeros([2]), "R_elbow": np.zeros([2]), "R_wrist": np.zeros([2]),
                                "L_pelvis": np.zeros([2]), "L_knee": np.zeros([2]), "L_ankle": np.zeros([2]),
                                "R_pelvis": np.zeros([2]), "R_knee": np.zeros([2]), "R_ankle": np.zeros([2]),
                                }
            
            bboxes= np.zeros([4,2])
            bbox_scores= np.zeros([4])
            
            ped_ids= defaultdict(lambda: deque(maxlen=video_info.total_frames))
            key_point_id= defaultdict(lambda: deque(maxlen=video_info.total_frames))
            
            
            next_id=0
            k=0
            while video.isOpened():
                success, frame= video.read()
                frame_idx+=1
                if not success:
                    break
                
                pose_est_results_last = pose_est_results
                (pred_3d_data_samples, pose_est_results, pose_est_results_list, pred_3d_instances,
                next_id, ped_ids, key_point_id) = process_one_image(
                    args=args,
                    detector=detector,
                    frame=frame,
                    frame_idx=frame_idx,
                    pose_estimator=pose_estimator,
                    pose_est_results_last=pose_est_results_last,
                    pose_est_results_list=pose_est_results_list,
                    next_id=next_id,
                    visualize_frame=mmcv.bgr2rgb(frame),
                    visualizer=visualizer,
                    bbox_id= ped_ids,
                    key_point_id= key_point_id
                )
                annotated_frame = frame.copy()
                cv2.polylines(annotated_frame, [Target_region], isClosed=True, color=(0, 255,0), thickness=5)
            
                black_region_A,black_region_B= get_img_coord(args)
                
                cv2.polylines(annotated_frame, [ black_region_A], isClosed=False, color=(0, 0, 0), thickness=5)
                cv2.polylines(annotated_frame, [ black_region_B], isClosed=False, color=(0, 0, 0), thickness=5)
                
                all_pedestrian_positions=[]
                all_pedestrian_score=[]
                for i,pose_est_result in enumerate(pose_est_results):
                    id= pose_est_result.track_id
                    #for pose in pose_est_results:
                    results= pred_3d_data_samples.pred_instances.transformed_keypoints[i]#.astype(int)
                    results_3d= pred_3d_data_samples.pred_instances.keypoints[i]
                    results_3d_save= pred_3d_data_samples.pred_instances.keypoints[i]
                    score= pred_3d_data_samples.pred_instances.keypoint_scores[i]
                    bboxes= pred_3d_data_samples.pred_instances.bboxes[i]
                    bbox_scores= pred_3d_data_samples.pred_instances.bbox_scores[i]
                    score= np.array(score, dtype=np.float32).copy()
                    results= np.array(results, dtype=np.float32).copy()
                    bboxes= np.array(bboxes, dtype=np.float32).copy()
                    results_3d_save= np.array(results_3d_save, dtype=np.float32).copy()
    
        
                
                    key_points_position['head']= results_3d[0]
                    for j in results_3d[23:91]:
                        key_points_position['head']+=j
                        key_points_position['head']/=(len(results_3d[23:91])+1)
                        key_points_position['L_shoulder']= results_3d[5]
                        key_points_position['L_elbow']= results_3d[7]
                        key_points_position['L_wrist']= results_3d[9]
                        key_points_position['R_shoulder']= results_3d[6]
                        key_points_position['R_elbow']= results_3d[8]
                        key_points_position['R_wrist']= results_3d[10]
                        key_points_position['L_pelvis']= results_3d[11]
                        key_points_position['L_knee']= results_3d[13]
                        key_points_position['L_ankle']= results_3d[15]
                        key_points_position['R_pelvis']= results_3d[12]
                        key_points_position['R_knee']= results_3d[14]
                        key_points_position['R_ankle']= results_3d[16]
                        key_points_position['M_shoulder']= (key_points_position['L_shoulder']+key_points_position['R_shoulder'])/2
                        key_points_position['M_pelvis']= (key_points_position['L_pelvis']+key_points_position['R_pelvis'])/2
                    
                    key_points_position_save['head']= results_3d_save[0]
                    for j in results_3d_save[23:91]:
                        key_points_position_save['head']+=j
                        key_points_position_save['head']/=(len(results_3d_save[23:91])+1)
                        key_points_position_save['L_shoulder']= results_3d_save[5]
                        key_points_position_save['L_elbow']= results_3d_save[7]
                        key_points_position_save['L_wrist']= results_3d_save[9]
                        key_points_position_save['R_shoulder']= results_3d_save[6]
                        key_points_position_save['R_elbow']= results_3d_save[8]
                        key_points_position_save['R_wrist']= results_3d_save[10]
                        key_points_position_save['L_pelvis']= results_3d_save[11]
                        key_points_position_save['L_knee']= results_3d_save[13]
                        key_points_position_save['L_ankle']= results_3d_save[15]
                        key_points_position_save['R_pelvis']= results_3d_save[12]
                        key_points_position_save['R_knee']= results_3d_save[14]
                        key_points_position_save['R_ankle']= results_3d_save[16]
                        key_points_position_save['M_shoulder']= (key_points_position_save['L_shoulder']+key_points_position_save['R_shoulder'])/2
                        key_points_position_save['M_pelvis']= (key_points_position_save['L_pelvis']+key_points_position_save['R_pelvis'])/2
                        key_points_score['head']= score[0]
                        key_points_score['L_shoulder']= score[5]
                        key_points_score['L_elbow']= score[7]
                        key_points_score['L_wrist']= score[9]
                        key_points_score['R_shoulder']= score[6]
                        key_points_score['R_elbow']= score[8]
                        key_points_score['R_wrist']= score[10]
                        key_points_score['L_pelvis']= score[11]
                        key_points_score['L_knee']= score[13]
                        key_points_score['L_ankle']= score[15]
                        key_points_score['R_pelvis']= score[12]
                        key_points_score['R_knee']= score[14]
                        key_points_score['R_ankle']= score[16]
                        key_points_score['M_shoulder']= (key_points_score['L_shoulder']+key_points_score['R_shoulder'])/2
                        key_points_score['M_pelvis']= (key_points_score['L_pelvis']+key_points_score['R_pelvis'])/2
                    
                    for j in results[23:91]:
                        key_points_position_2d['head']+=j
                        key_points_position_2d['head']/=(len(results[23:91])+1)
                        key_points_position_2d['L_shoulder']= results[5]
                        key_points_position_2d['L_elbow']= results[7]
                        key_points_position_2d['L_wrist']= results[9]
                        key_points_position_2d['R_shoulder']= results[6]
                        key_points_position_2d['R_elbow']= results[8]
                        key_points_position_2d['R_wrist']= results[10]
                        key_points_position_2d['L_pelvis']= results[11]
                        key_points_position_2d['L_knee']= results[13]
                        key_points_position_2d['L_ankle']= results[15]
                        key_points_position_2d['R_pelvis']= results[12]
                        key_points_position_2d['R_knee']= results[14]
                        key_points_position_2d['R_ankle']= results[16]
                        key_points_position_2d['M_shoulder']= (key_points_position_2d['L_shoulder']+key_points_position_2d['R_shoulder'])/2
                        key_points_position_2d['M_pelvis']= (key_points_position_2d['L_pelvis']+key_points_position_2d['R_pelvis'])/2
                    
                    
                    if is_point_in_polygon(key_points_position_2d["R_ankle"], Target_region) or is_point_in_polygon(key_points_position_2d["L_ankle"], Target_region):
                        annotated_frame = draw_keypoints_on_frame(annotated_frame, results, color=(0, 255, 0))
                        annotated_frame = draw_skeleton_on_frame(annotated_frame, results,ids=id, color=(255, 0, 0))
                        transformed_L_ankle_pixel= Mapping_real(key_points_position_2d['L_ankle'],args=args)
                        transformed_R_ankle_pixel= Mapping_real(key_points_position_2d['R_ankle'],args=args)
                        updated_point, degree= transformation(key_points_position, transformed_L_ankle_pixel, transformed_R_ankle_pixel,args=args)       
                        scaled_positions=scale_z_values(updated_point)
                        row_temp["frame_id"]= frame_idx
                        row_temp["ped_id"]= id
                        row_temp["key_points_scores"]= key_points_score
                        row_temp["key_points_positions"]= scaled_positions
                        row_temp["raw_key_points"]= key_points_position_save
                        row_temp["key_points_position_2ds"]= key_points_position_2d
                        row_temp["bboxes_2d"]= bboxes
                        row_temp["bbox_scores_2d"]= bbox_scores

                        row[k] = copy.deepcopy(row_temp)

                  
                        all_pedestrian_positions.append(scaled_positions)
                        all_pedestrian_score.append(key_points_score)
                        k+=1
                plot_img = plot_for_multiple_pedestrians(all_pedestrian_positions,all_pedestrian_score)
                plot_img_resized = cv2.resize(plot_img, (annotated_frame.shape[0], annotated_frame.shape[0]))
                plot_img_resized = cv2.cvtColor(plot_img_resized, cv2.COLOR_RGB2BGR)
                annotated_frame = np.hstack((annotated_frame, plot_img_resized))
                white_background = np.ones((3840, 7680, 3), dtype=np.uint8) * 255
                # annotated_frame을 중앙에 배치
                x_offset = (7680 - annotated_frame.shape[1]) // 2  # 좌우 여백
                y_offset = (3840 - annotated_frame.shape[0]) // 2  # 상하 여백
                # 하얀색 배경에 annotated_frame 복사
                white_background[y_offset:y_offset + annotated_frame.shape[0], x_offset:x_offset + annotated_frame.shape[1]] = annotated_frame
                if video_writer is None:
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(output_file, fourcc, fps,
                                                   (white_background.shape[1], white_background.shape[0]))
                video_writer.write(white_background)
                gc.collect()
                if args.show:
                    resized_frame = cv2.resize(white_background, (1920, 960))
                    cv2.imshow("frame", resized_frame) #
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            video.release()
            data_2= list(row.values())
            df = pd.DataFrame(data_2, columns=columns)
            df.to_csv(output_csv, index=False)
            print(f"Data saved to {output_csv}")
            if args.show:
                cv2.destroyAllWindows()
            if save_output:
                input_type = input_type.replace('webcam', 'video')
                print_log(
                    f'the output {input_type} has been saved at {output_file}',
                    logger='current',
                    level=logging.INFO)


if __name__ == '__main__':
    main()
