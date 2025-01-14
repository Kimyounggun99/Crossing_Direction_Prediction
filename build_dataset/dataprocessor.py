# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
from argparse import ArgumentParser


import numpy as np
import time
import torch
from matplotlib.colors import Normalize
import os 
import pandas as pd
import glob
import json
import ast
import re
import torch
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--bs", default=4, type=int
    )
    parser.add_argument(
        "--epoch", default=100, type=int
    )
    parser.add_argument(
        "--lr", default=0.0001, type=int
    )
    parser.add_argument(
        "--interval", default=50, type=int
    )
    parser.add_argument(
        "--model", default='Transformer_based_model', type=str
    )
    parser.add_argument(
        "--mode", default='train', type=str
    )
    parser.add_argument(
        "--experiment-type", default='TTE', type=str, help='TTE/SEQ'
    )
    parser.add_argument(
        "--traj_type", default="cartesian", type=str
    )
    parser.add_argument(
        "--raw_point", default=False, type=bool
    )
    parser.add_argument(
        "--observation-time", default=4, type=int, help='Observation_time'
    )
    parser.add_argument(
        "--input_dir", default="/data/kim/CCTV_data", type=str
    )
    
    parser.add_argument(
        "--output_dir", default="./logs", type=str
    )
    parser.add_argument(
        "--save_dir", default="./checkpoints", type=str
    )
    parser.add_argument(
        "--load", default=None, type=str
    )
    args = parser.parse_args()
    return args


def safe_eval(value):
    """Safely evaluates a string into a Python object."""
    try:
        if pd.notna(value):
            # 이미 딕셔너리, 리스트, 또는 numpy 배열인 경우
            if isinstance(value, (dict, list, np.ndarray)):
                return value
            # 문자열인 경우 파싱
            if isinstance(value, str):
        
                # 'array(...)'를 numpy 형식으로 변환
                if 'array' in value:
                    # dtype 제거
                    value = re.sub(r', dtype=.*?\)', ')', value)
                    # array를 np.array로 대체
                    value = value.replace('array', 'np.array')
    
                    return eval(value, {"np": np})

 
            if value[0]=='[':
                
                inside_brackets = re.search(r'\[(.*?)\]', value)
                extracted_content = inside_brackets.group(1)
                numbers = list(map(float, extracted_content.split()))
                return numbers
    except (ValueError, SyntaxError, NameError) as e:
  
        print(f"Malformed value: {value}, Error: {e}")
    
        return None
    return value

def dict_to_array(column_data):
    """Convert dictionary-like column to a numpy array."""
    valid_data = column_data.dropna().reset_index(drop=True)
    if len(valid_data) == 0:
        print(f"Column {column_data.name} has no valid data.")
        return None

    first_valid = valid_data.iloc[0]

    # 딕셔너리 형태 처리
    if isinstance(first_valid, dict):
        keys = list(first_valid.keys())
        data_array = np.array([
            [np.array(d.get(key, np.nan)) for key in keys]
            for d in valid_data
        ])
        
        return data_array

    parsed_data = [ast.literal_eval(row) for row in valid_data]
    values = [list(row.values()) for row in parsed_data]
    array_data = np.array(values)  # (data_num, feature_num)
    array_data = np.expand_dims(array_data, axis=-1)  # (data_num, feature_num, 1)
  

    return array_data



def process_csv_to_arrays(file_path):
    df = pd.read_csv(file_path)
    
    # Columns with JSON-like or dictionary data
    key_columns = ['key_points_scores', 'key_points_positions', 'raw_key_points', 'key_points_position_2ds']
    array_columns = ['bboxes_2d']
    
    # Evaluate JSON-like columns
   
    for col in key_columns + array_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_eval)
    
    # Convert columns to arrays
    arrays = {}
    for col in key_columns:
        if col in df.columns:
            try:
                arrays[col] = dict_to_array(df[col])
            except ValueError as e:
                print(f"Skipping column {col}: {e}")
    
    for col in array_columns:
        if col in df.columns:
            arrays[col] = np.array(df[col].dropna().tolist())
    
    # Add non-key columns
    non_key_columns = [col for col in df.columns if col not in key_columns + array_columns]
    
    for col in non_key_columns:
        arrays[col] = df[col].values
    
    return arrays

# 데이터 구조 확인
def inspect_data(df, columns):
    for col in columns:
        if col in df.columns:
            print(f"Column: {col}")
            for i, value in enumerate(df[col].dropna().head(5)):  # 첫 5개 값 확인
                print(f"Row {i}: {type(value)}, {value}")


def get_center_and_polar_angles(data):

    R_ankle= data[:,-1]
    L_ankle= data[:,-4]
    center= (R_ankle[:,:2]+L_ankle[:,:2])/2. #(num, 2)
    crosswalk_A= np.zeros_like(center)
    crosswalk_A[:,0]+=1
    norm_crosswalk_A = np.linalg.norm(crosswalk_A, axis=1, keepdims=True)
    norm_center = np.linalg.norm(center, axis=1, keepdims=True) 
    # 2. 코사인 값 계산
    dot_product = np.sum(crosswalk_A * center, axis=1, keepdims=True)
    cosine_values = dot_product / (norm_crosswalk_A * norm_center)
    # 3. 사인 값 계산
    cross_product = np.cross(crosswalk_A, center)
    sine_values = cross_product / (norm_crosswalk_A.flatten() * norm_center.flatten())
    # 4. 결과 배열 생성
    polar_angles = np.hstack((cosine_values, sine_values.reshape(-1, 1)))


    return center, polar_angles


def make_temperal_date(data, observation_frames=24, overlap=0.5):
    
    ped_ids= np.unique(data['ped_id'])

    useful_ids= []
    all_data= {}

    j=0
    for id in ped_ids:
        data_id_len= len(np.where(data['ped_id']==id)[0])
        
        if data_id_len<observation_frames:
            continue
        
        useful_ids.append(id)
        data_id_len2= data_id_len-observation_frames
        inc_num= int(observation_frames*overlap)
        sample_num= (data_id_len2//inc_num)

        start_idx= (data_id_len-observation_frames)%inc_num


        #frame_id= data['frame_id'][np.where(data['ped_id']==id)[0]]
        key_points_scores= data['key_points_scores'][np.where(data['ped_id']==id)[0]]
        key_points_positions= data['key_points_positions'][np.where(data['ped_id']==id)[0]]
        raw_key_points= data['raw_key_points'][np.where(data['ped_id']==id)[0]]
        key_points_position_2ds= data['key_points_position_2ds'][np.where(data['ped_id']==id)[0]]
        bboxes_2d= data['bboxes_2d'][np.where(data['ped_id']==id)[0]]
        bbox_scores_2d= data['bbox_scores_2d'][np.where(data['ped_id']==id)[0]]

        C_labels= data['C_label'][np.where(data['ped_id']==id)[0]]
        J_labels= data['J_label'][np.where(data['ped_id']==id)[0]]


        center, polar_angles= get_center_and_polar_angles(key_points_positions)



        current_idx= start_idx
        
        key_points_scores_list= []
        key_points_positions_list= []
        raw_key_points_list= []
        key_points_position_2ds_list= []
        bboxes_2d_list= []
        bbox_scores_2d_list= []

        C_labels_list= []
        J_labels_list= []

        center_list= []
        polar_angles_list= []
        velocity_list=[]
        for i in range(sample_num+1):
            key_points_scores_list.append(key_points_scores[current_idx:current_idx+observation_frames])
            key_points_positions_list.append(key_points_positions[current_idx:current_idx+observation_frames])
            raw_key_points_list.append(raw_key_points[current_idx:current_idx+observation_frames])
            key_points_position_2ds_list.append(key_points_position_2ds[current_idx:current_idx+observation_frames])
       
            bboxes_2d_list.append(bboxes_2d[current_idx:current_idx+observation_frames])

            bbox_scores_2d_list.append(bbox_scores_2d[current_idx:current_idx+observation_frames])
            
            C_labels_list.append(C_labels[current_idx:current_idx+observation_frames])
            J_labels_list.append(J_labels[current_idx:current_idx+observation_frames])

            center_list.append(center[current_idx:current_idx+observation_frames])

            delta_center= center[current_idx:current_idx+observation_frames][0]-center[current_idx:current_idx+observation_frames][-1]

            vel= np.sqrt(delta_center[0]**2+delta_center[1]**2)
            velocity= np.array([vel])
            velocity_list.append(velocity.repeat(len(center[current_idx:current_idx+observation_frames])).reshape(-1,1))

            polar_angles_list.append(polar_angles[current_idx:current_idx+observation_frames])

            current_idx+=inc_num

        key_points_scores=np.array(key_points_scores_list).reshape(len(key_points_scores_list),len(key_points_scores_list[0]), len(key_points_scores_list[0][0]),len(key_points_scores_list[0][0][0]))
        key_points_positions=np.array(key_points_positions_list).reshape(len(key_points_positions_list),len(key_points_positions_list[0]), len(key_points_positions_list[0][0]),len(key_points_positions_list[0][0][0]))
        raw_key_points=np.array(raw_key_points_list).reshape(len(raw_key_points_list),len(raw_key_points_list[0]), len(raw_key_points_list[0][0]),len(raw_key_points_list[0][0][0]))
        key_points_position_2ds=np.array(key_points_position_2ds_list).reshape(len(key_points_position_2ds_list),len(key_points_position_2ds_list[0]), len(key_points_position_2ds_list[0][0]),len(key_points_position_2ds_list[0][0][0]))
        bboxes_2d=np.array(bboxes_2d_list).reshape(len(bboxes_2d_list),len(bboxes_2d_list[0]), len(bboxes_2d_list[0][0]))
        bbox_scores_2d=np.array(bbox_scores_2d_list).reshape(len(bbox_scores_2d_list),len(bbox_scores_2d_list[0]))
        
        C_labels=np.array(C_labels_list).reshape(len(C_labels_list),len(C_labels_list[0]))
        J_labels=np.array(J_labels_list).reshape(len(J_labels_list),len(J_labels_list[0]))
  
        velocity= np.array(velocity_list).reshape(len(velocity_list),len(velocity_list[0]),len(velocity_list[0][0]))

        center=np.array(center_list).reshape(len(center_list),len(center_list[0]), len(center_list[0][0]))
        polar_angles=np.array(polar_angles_list).reshape(len(polar_angles_list),len(polar_angles_list[0]), len(polar_angles_list[0][0]))
       
        assert len(np.unique(C_labels))<2, "Wrong labels!!!!!!!!!!!!!!"
        
        C_labels= C_labels[:,0]
        J_labels= J_labels[:,0]
        
        
        if j==0:
            all_data['key_points_scores']=key_points_scores
            all_data['key_points_positions']=key_points_positions
            all_data['raw_key_points']=raw_key_points
            all_data['key_points_position_2ds']=key_points_position_2ds
            all_data['bboxes_2d']=bboxes_2d
            all_data['bbox_scores_2d']=bbox_scores_2d
            all_data['C_labels']=C_labels
            all_data['J_labels']=J_labels

            all_data['center']=center
            all_data['polar_angles']=polar_angles
            all_data['velocity']=velocity
            

            j+=1
        else:
            all_data['key_points_scores']=np.concatenate((all_data['key_points_scores'], key_points_scores),axis=0)
            all_data['key_points_positions']=np.concatenate((all_data['key_points_positions'], key_points_positions),axis=0)
            all_data['raw_key_points']=np.concatenate((all_data['raw_key_points'], raw_key_points),axis=0)
            all_data['key_points_position_2ds']=np.concatenate((all_data['key_points_position_2ds'], key_points_position_2ds),axis=0)
            all_data['bboxes_2d']=np.concatenate((all_data['bboxes_2d'], bboxes_2d),axis=0)
            all_data['bbox_scores_2d']=np.concatenate((all_data['bbox_scores_2d'], bbox_scores_2d),axis=0)

            all_data['C_labels']=np.concatenate((all_data['C_labels'], C_labels),axis=0)
            all_data['J_labels']=np.concatenate((all_data['J_labels'], J_labels),axis=0)

            all_data['center']=np.concatenate((all_data['center'], center),axis=0)
            all_data['polar_angles']=np.concatenate((all_data['polar_angles'], polar_angles),axis=0)
            all_data['velocity']=np.concatenate((all_data['velocity'], velocity),axis=0)

    
    return all_data




def make_raw_temperal_date(data, observation_frames=24, overlap=0.5):
    
    ped_ids= np.unique(data['ped_id'])

    useful_ids= []
    all_data= {}

    j=0
    for id in ped_ids:
        data_id_len= len(np.where(data['ped_id']==id)[0])
        
        if data_id_len<observation_frames:
            continue
        
        useful_ids.append(id)
        data_id_len2= data_id_len-observation_frames
        inc_num= int(observation_frames*overlap)
        sample_num= (data_id_len2//inc_num)

        start_idx= (data_id_len-observation_frames)%inc_num


        #frame_id= data['frame_id'][np.where(data['ped_id']==id)[0]]
        key_points_scores= data['key_points_scores'][np.where(data['ped_id']==id)[0]]
        key_points_positions= data['key_points_positions'][np.where(data['ped_id']==id)[0]]
        raw_key_points= data['raw_key_points'][np.where(data['ped_id']==id)[0]]
        key_points_position_2ds= data['key_points_position_2ds'][np.where(data['ped_id']==id)[0]]
        bboxes_2d= data['bboxes_2d'][np.where(data['ped_id']==id)[0]]
        bbox_scores_2d= data['bbox_scores_2d'][np.where(data['ped_id']==id)[0]]

        C_labels= data['C_label'][np.where(data['ped_id']==id)[0]]
        J_labels= data['J_label'][np.where(data['ped_id']==id)[0]]


        center, polar_angles= get_center_and_polar_angles(raw_key_points)



        current_idx= start_idx
        
        key_points_scores_list= []
        key_points_positions_list= []
        raw_key_points_list= []
        key_points_position_2ds_list= []
        bboxes_2d_list= []
        bbox_scores_2d_list= []

        C_labels_list= []
        J_labels_list= []

        center_list= []
        polar_angles_list= []
        velocity_list=[]
        for i in range(sample_num+1):
            key_points_scores_list.append(key_points_scores[current_idx:current_idx+observation_frames])
            key_points_positions_list.append(key_points_positions[current_idx:current_idx+observation_frames])
            raw_key_points_list.append(raw_key_points[current_idx:current_idx+observation_frames])
            key_points_position_2ds_list.append(key_points_position_2ds[current_idx:current_idx+observation_frames])
       
            bboxes_2d_list.append(bboxes_2d[current_idx:current_idx+observation_frames])

            bbox_scores_2d_list.append(bbox_scores_2d[current_idx:current_idx+observation_frames])
            
            C_labels_list.append(C_labels[current_idx:current_idx+observation_frames])
            J_labels_list.append(J_labels[current_idx:current_idx+observation_frames])

            center_list.append(center[current_idx:current_idx+observation_frames])

            delta_center= center[current_idx:current_idx+observation_frames][0]-center[current_idx:current_idx+observation_frames][-1]

            vel= np.sqrt(delta_center[0]**2+delta_center[1]**2)
            velocity= np.array([vel])
            velocity_list.append(velocity.repeat(len(center[current_idx:current_idx+observation_frames])).reshape(-1,1))

            polar_angles_list.append(polar_angles[current_idx:current_idx+observation_frames])

            current_idx+=inc_num

        key_points_scores=np.array(key_points_scores_list).reshape(len(key_points_scores_list),len(key_points_scores_list[0]), len(key_points_scores_list[0][0]),len(key_points_scores_list[0][0][0]))
        key_points_positions=np.array(key_points_positions_list).reshape(len(key_points_positions_list),len(key_points_positions_list[0]), len(key_points_positions_list[0][0]),len(key_points_positions_list[0][0][0]))
        raw_key_points=np.array(raw_key_points_list).reshape(len(raw_key_points_list),len(raw_key_points_list[0]), len(raw_key_points_list[0][0]),len(raw_key_points_list[0][0][0]))
        key_points_position_2ds=np.array(key_points_position_2ds_list).reshape(len(key_points_position_2ds_list),len(key_points_position_2ds_list[0]), len(key_points_position_2ds_list[0][0]),len(key_points_position_2ds_list[0][0][0]))
        bboxes_2d=np.array(bboxes_2d_list).reshape(len(bboxes_2d_list),len(bboxes_2d_list[0]), len(bboxes_2d_list[0][0]))
        bbox_scores_2d=np.array(bbox_scores_2d_list).reshape(len(bbox_scores_2d_list),len(bbox_scores_2d_list[0]))
        
        C_labels=np.array(C_labels_list).reshape(len(C_labels_list),len(C_labels_list[0]))
        J_labels=np.array(J_labels_list).reshape(len(J_labels_list),len(J_labels_list[0]))
  
        velocity= np.array(velocity_list).reshape(len(velocity_list),len(velocity_list[0]),len(velocity_list[0][0]))

        center=np.array(center_list).reshape(len(center_list),len(center_list[0]), len(center_list[0][0]))
        polar_angles=np.array(polar_angles_list).reshape(len(polar_angles_list),len(polar_angles_list[0]), len(polar_angles_list[0][0]))
       
        assert len(np.unique(C_labels))<2, "Wrong labels!!!!!!!!!!!!!!"
        
        C_labels= C_labels[:,0]
        J_labels= J_labels[:,0]
        
        
        if j==0:
            all_data['key_points_scores']=key_points_scores
            all_data['key_points_positions']=key_points_positions
            all_data['raw_key_points']=raw_key_points
            all_data['key_points_position_2ds']=key_points_position_2ds
            all_data['bboxes_2d']=bboxes_2d
            all_data['bbox_scores_2d']=bbox_scores_2d
            all_data['C_labels']=C_labels
            all_data['J_labels']=J_labels

            all_data['center']=center
            all_data['polar_angles']=polar_angles
            all_data['velocity']=velocity
            

            j+=1
        else:
            all_data['key_points_scores']=np.concatenate((all_data['key_points_scores'], key_points_scores),axis=0)
            all_data['key_points_positions']=np.concatenate((all_data['key_points_positions'], key_points_positions),axis=0)
            all_data['raw_key_points']=np.concatenate((all_data['raw_key_points'], raw_key_points),axis=0)
            all_data['key_points_position_2ds']=np.concatenate((all_data['key_points_position_2ds'], key_points_position_2ds),axis=0)
            all_data['bboxes_2d']=np.concatenate((all_data['bboxes_2d'], bboxes_2d),axis=0)
            all_data['bbox_scores_2d']=np.concatenate((all_data['bbox_scores_2d'], bbox_scores_2d),axis=0)

            all_data['C_labels']=np.concatenate((all_data['C_labels'], C_labels),axis=0)
            all_data['J_labels']=np.concatenate((all_data['J_labels'], J_labels),axis=0)

            all_data['center']=np.concatenate((all_data['center'], center),axis=0)
            all_data['polar_angles']=np.concatenate((all_data['polar_angles'], polar_angles),axis=0)
            all_data['velocity']=np.concatenate((all_data['velocity'], velocity),axis=0)

    
    return all_data









def make_time_to_event(data, observation_time):
    
    observation_frames= observation_time*24

    ped_ids= np.unique(data['ped_id'])

    useful_ids= []
    all_data= {}

    j=0
    for id in ped_ids:
        data_id_len= len(np.where(data['ped_id']==id)[0])
        
        #if data_id_len<24:
        #        continue
        
        key_points_scores= data['key_points_scores'][np.where(data['ped_id']==id)[0]]
        key_points_positions= data['key_points_positions'][np.where(data['ped_id']==id)[0]]
        raw_key_points= data['raw_key_points'][np.where(data['ped_id']==id)[0]]
        key_points_position_2ds= data['key_points_position_2ds'][np.where(data['ped_id']==id)[0]]
        bboxes_2d= data['bboxes_2d'][np.where(data['ped_id']==id)[0]]
        bbox_scores_2d= data['bbox_scores_2d'][np.where(data['ped_id']==id)[0]]
        
        C_labels= data['C_label'][np.where(data['ped_id']==id)[0]]
        J_labels= data['J_label'][np.where(data['ped_id']==id)[0]]
        if len(np.unique(C_labels))>1:
            breakpoint()

        if data_id_len<observation_frames:
            lack_frames= observation_frames- data_id_len
            for frame in range(lack_frames):
                key_points_scores= np.concatenate((key_points_scores, key_points_scores[-1].reshape(1, key_points_scores.shape[1], key_points_scores.shape[2])), axis=0)
                key_points_positions= np.concatenate((key_points_positions, key_points_positions[-1].reshape(1, key_points_positions.shape[1], key_points_positions.shape[2])), axis=0)
                raw_key_points= np.concatenate((raw_key_points, raw_key_points[-1].reshape(1, raw_key_points.shape[1], raw_key_points.shape[2])), axis=0)
                key_points_position_2ds= np.concatenate((key_points_position_2ds, key_points_position_2ds[-1].reshape(1, key_points_position_2ds.shape[1], key_points_position_2ds.shape[2])), axis=0)
             
                bboxes_2d= np.concatenate((bboxes_2d, bboxes_2d[-1].reshape(1, bboxes_2d.shape[1])), axis=0)
                bbox_scores_2d= np.concatenate((bbox_scores_2d, bbox_scores_2d[-1].reshape(1,)), axis=0)
                #C_labels= np.concatenate((C_labels, C_labels[-1].reshape(1,)), axis=0)
                #J_labels= np.concatenate((J_labels, J_labels[-1].reshape(1,)), axis=0)

        else:    
            key_points_scores= key_points_scores[-observation_frames:]
            key_points_positions= key_points_positions[-observation_frames:]
            raw_key_points= raw_key_points[-observation_frames:]
            key_points_position_2ds= key_points_position_2ds[-observation_frames:]
            bboxes_2d= bboxes_2d[-observation_frames:]
            bbox_scores_2d= bbox_scores_2d[-observation_frames:]
            #C_labels= C_labels[-observation_frames:]
            #J_labels= J_labels[-observation_frames:]
        
        center, polar_angles= get_center_and_polar_angles(key_points_positions)

        key_points_scores= key_points_scores.reshape(1, key_points_scores.shape[0], key_points_scores.shape[1], key_points_scores.shape[2])
        key_points_positions= key_points_positions.reshape(1, key_points_positions.shape[0], key_points_positions.shape[1], key_points_positions.shape[2])
        raw_key_points= raw_key_points.reshape(1, raw_key_points.shape[0], raw_key_points.shape[1], raw_key_points.shape[2])
        key_points_position_2ds= key_points_position_2ds.reshape(1, key_points_position_2ds.shape[0], key_points_position_2ds.shape[1], key_points_position_2ds.shape[2])
        bboxes_2d= bboxes_2d.reshape(1, bboxes_2d.shape[0], bboxes_2d.shape[1])
        bbox_scores_2d= bbox_scores_2d.reshape(1,-1)
        C_labels= np.unique(C_labels).reshape(1,-1)
        J_labels= np.unique(J_labels).reshape(1,-1)
        
        center= center.reshape(1, center.shape[0], center.shape[1])
        polar_angles= polar_angles.reshape(1, polar_angles.shape[0], polar_angles.shape[1])
        if j==0:
            all_data['key_points_scores']=key_points_scores
            all_data['key_points_positions']=key_points_positions
            all_data['raw_key_points']=raw_key_points
            all_data['key_points_position_2ds']=key_points_position_2ds
            all_data['bboxes_2d']=bboxes_2d
            all_data['bbox_scores_2d']=bbox_scores_2d
            all_data['C_labels']=C_labels
            all_data['J_labels']=J_labels

            all_data['center']=center
            all_data['polar_angles']=polar_angles
            all_data['C_labels']= C_labels
            j+=1
        else:
            all_data['key_points_scores']=np.concatenate((all_data['key_points_scores'], key_points_scores),axis=0)
            all_data['key_points_positions']=np.concatenate((all_data['key_points_positions'], key_points_positions),axis=0)
            all_data['raw_key_points']=np.concatenate((all_data['raw_key_points'], raw_key_points),axis=0)
            all_data['key_points_position_2ds']=np.concatenate((all_data['key_points_position_2ds'], key_points_position_2ds),axis=0)
            all_data['bboxes_2d']=np.concatenate((all_data['bboxes_2d'], bboxes_2d),axis=0)
            all_data['bbox_scores_2d']=np.concatenate((all_data['bbox_scores_2d'], bbox_scores_2d),axis=0)

            all_data['C_labels']=np.concatenate((all_data['C_labels'], C_labels),axis=0)
            all_data['J_labels']=np.concatenate((all_data['J_labels'], J_labels),axis=0)

            all_data['center']=np.concatenate((all_data['center'], center),axis=0)
            all_data['polar_angles']=np.concatenate((all_data['polar_angles'], polar_angles),axis=0)
    
    return all_data










def build_data(args):
    print('Building Dataset...')
    if args.input_dir==None:
        raise AssertionError("Insert input path!!!")
    else:
        data_path= args.input_dir
        
    train_path= data_path+'/train'
    test_path= data_path+'/test'


    train_files = glob.glob(os.path.join(train_path, "*.csv"))
    test_files= glob.glob(os.path.join(test_path, "*.csv"))
    
    
    train_all_datas= {}
    for idx, train_file in enumerate(train_files):
      

        train_data= process_csv_to_arrays(train_file)
        
        if args.experiment_type == 'TTE':
            train_data= make_time_to_event(train_data, observation_time=args.observation_time)
            
        elif args.experiment_type == 'SEQ':
            if args.raw_point==False:
                train_data= make_temperal_date(train_data, observation_frames=args.observation_time*24)
            else:
                train_data= make_raw_temperal_date(train_data, observation_frames=args.observation_time*24)
            
            
        else:
            breakpoint()


        if idx==0:
            train_all_datas= train_data
        else:
            for key, value in train_data.items():
                train_all_datas[key]= np.concatenate((train_all_datas[key], train_data[key]), axis=0)
     
    
    test_all_datas= {}
    for idx, test_file in enumerate(test_files):
        
        test_data= process_csv_to_arrays(test_file)
        
     
        if args.experiment_type == 'TTE':
            test_data= make_time_to_event(test_data, observation_time=args.observation_time)
        elif args.experiment_type == 'SEQ':
            if args.raw_point==False:
                test_data= make_temperal_date(test_data, observation_frames=args.observation_time*24) 
            else:
                test_data= make_raw_temperal_date(test_data, observation_frames=args.observation_time*24) 
        else:
            breakpoint()

        if idx==0:
            test_all_datas= test_data
        else:
            for key, value in test_data.items():
                test_all_datas[key]= np.concatenate((test_all_datas[key], test_data[key]), axis=0)


    if args.experiment_type=='TTE':
        train_all_datas['C_labels']= train_all_datas['C_labels'].reshape(-1)
        test_all_datas['C_labels']= test_all_datas['C_labels'].reshape(-1)
    
    print(f'\n***********************\ Dataset Type: {args.experiment_type} ***********************\n')
    print(f"Total number of trainset: {len(train_all_datas['C_labels'])}")
    print(f"The number of A of trainset: {len(np.where(train_all_datas['C_labels']==0)[0])}")
    print(f"The number of B of trainset: {len(np.where(train_all_datas['C_labels']==1)[0])}")
    print('\n')
    print(f"Total number of testset: {len(test_all_datas['C_labels'])}")
    print(f"The number of A of testset: {len(np.where(test_all_datas['C_labels']==0)[0])}")
    print(f"The number of B of testset: {len(np.where(test_all_datas['C_labels']==1)[0])}")
    print('\n**********************************************\n')
    print("Every dataset has been uploaded!! ")
    return train_all_datas, test_all_datas
   

class get_data(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): Preprocessed data containing multiple keys.
        """
        self.data = data
        self.length = data['C_labels'].shape[0]  # Number of samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns a single sample as a dictionary.
        """
        sample = {
            'key_points_scores': self.data['key_points_scores'][idx],
            'key_points_positions': self.data['key_points_positions'][idx],
            'raw_key_points': self.data['raw_key_points'][idx],
            'key_points_position_2ds': self.data['key_points_position_2ds'][idx],
            'bboxes_2d': self.data['bboxes_2d'][idx],
            'bbox_scores_2d': self.data['bbox_scores_2d'][idx],
            'C_labels': self.data['C_labels'][idx],
            'J_labels': self.data['J_labels'][idx],
            'center': self.data['center'][idx],
            'polar_angles': self.data['polar_angles'][idx],

        }
        return sample



def create_dataloaders(train_data, test_data, batch_size=32, shuffle=True):
    """
    Create DataLoaders for training and testing datasets.

    Args:
        train_data (dict): Training data.
        test_data (dict): Testing data.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        train_loader, test_loader: PyTorch DataLoaders.
    """
    train_dataset = get_data(train_data)
    test_dataset = get_data(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,  drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print("All data were successfuly uploaded!!")
    return train_loader, test_loader

def main():
    args= parse_args()
    build_data(args)

    

if __name__ == '__main__':
    main()
