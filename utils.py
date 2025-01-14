import numpy as np

def get_img_coord(args):
    if args.cam_num=='orange_view1': 
        p1 = np.array([1907,115])  # P1: 기준점 (원점)
        p2 = np.array([1864,487])  # P2: X축
        p3 = np.array([2790,199]) # P3: Y축
    elif args.cam_num=='orange_view2':        
        # P1, P2, P3 정의
        p1 = np.array([1507, 524])  # P1: 기준점 (원점)
        p2 = np.array([1700, 1225])  # P2: X축
        p3 = np.array([2550, 407]) # P3: Y축
    elif args.cam_num=='orange_view3':
        p1 = np.array([2513, 314])  # P1: 기준점 (원점)
        p2 = np.array([2490, 444])  # P2: X축
        p3 = np.array([1319, 227]) # P3: Y축
    elif args.cam_num=='orange_view4':
        p1 = np.array([1218, 799])  # P1: 기준점 (원점)
        p2 = np.array([1331, 1257])  # P2: X축
        p3 = np.array([2308, 808]) # P3: Y축

    elif args.cam_num=='seminole_view1':
        p1 = np.array([1501,853])  # P1: 기준점 (원점)
        p2 = np.array([1432,1164])  # P2: X축
        p3 = np.array([2772,902]) # P3: Y축
    elif args.cam_num=='seminole_view2':
        p1 = np.array([1994, 522])  # P1: 기준점 (원점)
        p2 = np.array([2307,755])  # P2: X축
        p3 = np.array([1550, 582]) # P3: Y축
    elif args.cam_num=='seminole_view3':
        p1 = np.array([1870, 839])  # P1: 기준점 (원점)
        p2 = np.array([2046, 1450])  # P2: X축
        p3 = np.array([2608, 790]) # P3: Y축
    elif args.cam_num=='seminole_view4':

        p1 = np.array([1732,761])  # P1: 기준점 (원점)
        p2 = np.array([1685,1015])  # P2: X축
        p3 = np.array([1201, 741]) # P3: Y축


    elif args.cam_num=='orange_view1_sideview':
        p1 = np.array([2580,1925])  # P1: 기준점 (원점)
        p2 = np.array([2281,811])  # P2: X축
        p3 = np.array([239,1570]) # P3: Y축

    elif args.cam_num=='orange_view2_sideview':
        p1 = np.array([2302,1578])  # P1: 기준점 (원점)
        p2 = np.array([2219,914])  # P2: X축
        p3 = np.array([792,1160]) # P3: Y축

    p_M= p2+p3-p1
    black_region_A = np.concatenate((p1,p2), axis=0).reshape(2,2)
    black_region_B = np.concatenate((p1,p3), axis=0).reshape(2,2)


    return black_region_A,black_region_B


def Mapping_real(base_point, args):

    if args.cam_num=='orange_view1': #다시
        p1 = np.array([1907,115])  # P1: 기준점 (원점)
        p2 = np.array([1864,487])  # P2: X축
        p3 = np.array([2790,199]) # P3: Y축
    elif args.cam_num=='orange_view2':        
        # P1, P2, P3 정의
        p1 = np.array([1507, 524])  # P1: 기준점 (원점)
        p2 = np.array([1700, 1225])  # P2: X축
        p3 = np.array([2550, 407]) # P3: Y축
    elif args.cam_num=='orange_view3':
        p1 = np.array([2513, 314])  # P1: 기준점 (원점)
        p2 = np.array([2490, 444])  # P2: X축
        p3 = np.array([1319, 227]) # P3: Y축
    elif args.cam_num=='orange_view4':
        p1 = np.array([1218, 799])  # P1: 기준점 (원점)
        p2 = np.array([1331, 1257])  # P2: X축
        p3 = np.array([2308, 808]) # P3: Y축

    elif args.cam_num=='seminole_view1':
        p1 = np.array([1501,853])  # P1: 기준점 (원점)
        p2 = np.array([1432,1164])  # P2: X축
        p3 = np.array([2772,902]) # P3: Y축
    elif args.cam_num=='seminole_view2':
        p1 = np.array([1994, 522])  # P1: 기준점 (원점)
        p2 = np.array([2307,755])  # P2: X축
        p3 = np.array([1550, 582]) # P3: Y축
    elif args.cam_num=='seminole_view3':
        p1 = np.array([1870, 839])  # P1: 기준점 (원점)
        p2 = np.array([2046, 1450])  # P2: X축
        p3 = np.array([2608, 790]) # P3: Y축
    elif args.cam_num=='seminole_view4':

        p1 = np.array([1732,761])  # P1: 기준점 (원점)
        p2 = np.array([1685,1015])  # P2: X축
        p3 = np.array([1201, 741]) # P3: Y축


    elif args.cam_num=='orange_view1_sideview':
        p1 = np.array([2580,1925])  # P1: 기준점 (원점)
        p2 = np.array([2281,811])  # P2: X축
        p3 = np.array([239,1570]) # P3: Y축

    elif args.cam_num=='orange_view2_sideview':
        p1 = np.array([2302,1578])  # P1: 기준점 (원점)
        p2 = np.array([2219,914])  # P2: X축
        p3 = np.array([792,1160]) # P3: Y축


    # 단위 벡터 v1(p1→p3), v2(p1→p2)
    v1 = (p3 - p1) / np.linalg.norm(p3 - p1)
    v2 = (p2 - p1) / np.linalg.norm(p2 - p1)


       
    base_x, base_y = calculate_intersections(base_point, p1, v1, v2)
    transformed_base = calculate_ground_truth(base_x, base_y, p1, p2, p3)

    return transformed_base


def transformation(updated_point, transformed_L_ankle_pixel, transformed_R_ankle_pixel, args=None):
    """
    L_ankle과 R_ankle 벡터의 각도 차이만큼 updated_point의 모든 x, y 좌표를 회전시킴.
    
    Args:
        updated_point (dict): 원본 keypoints 좌표.
        transformed_L_ankle_pixel (array): 변환된 L_ankle 좌표.
        transformed_R_ankle_pixel (array): 변환된 R_ankle 좌표.

    Returns:
        dict: 회전된 keypoints 좌표.
    """
    c_x=0
    c_y=0
    i=0
    # 4. 중심 좌표 c1, c2 계산
    for  key, value in updated_point.items():
        x, y, z = value
        c_x+=x
        c_y+=y
        i+=1

    c_x/=i
    c_y/=i
    c1 = np.array([c_x, c_y])
    c2 = (transformed_L_ankle_pixel + transformed_R_ankle_pixel) / 2

    vector_c1 = c1  # 원점(0,0)에서 c1까지의 벡터
    vector_c2 = c2  # 원점(0,0)에서 c2까지의 벡터

    # 두 벡터 사이의 각도 계산
    dot_product = np.dot(vector_c1, vector_c2)
    norm_product = np.linalg.norm(vector_c1) * np.linalg.norm(vector_c2)
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    theta = np.arccos(cos_theta)


    if args.cam_num== 'seminole_view2' or args.cam_num=='seminole_view4':  
        if theta>0:
            theta-=1.2
        else:
            theta+=1.2
    else:
        if theta>0:
            theta-=0.6
        else:
            theta+=0.6

    
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # 5. 모든 포인트에 회전 행렬 적용
    rotated_points = {}
    for key, value in updated_point.items():
        x, y, z = value
        rotated_xy = np.dot(rotation_matrix, [x, y])
        rotated_points[key] = np.array([rotated_xy[0], rotated_xy[1], z], dtype=np.float32)

    
    c_x=0
    c_y=0
    i=0
    # 4. 중심 좌표 c1, c2 계산
    for  key, value in rotated_points.items():
        x, y, z = value
        c_x+=x
        c_y+=y
        i+=1

    c_x/=i
    c_y/=i
    c1 = np.array([c_x, c_y])
    c2 = (transformed_L_ankle_pixel + transformed_R_ankle_pixel) / 2

    scaled_points = {}
    

    for key, value in rotated_points.items():
        x, y, z = value
        scaled_x = (x - c1[0]) * 0.002 
        scaled_y = (y - c1[1]) * 0.002

        scaled_points[key] = np.array([scaled_x,scaled_y, z])
    if args.cam_num!= 'seminole_view2' and args.cam_num!='seminole_view4' and args.cam_num!='orange_view1_sideview' and args.cam_num!='orange_view2_sideview' :      
        scaled_points= rotate_around_z_180(scaled_points)

    if args.cam_num=='orange_view1_sideview':
        scaled_points= rotate_around_z_90(scaled_points, clockwise=True)
        scaled_points= rotate_around_z_90(scaled_points, clockwise=True)

    
    scaled_points2= {}
    for key, value in scaled_points.items():
        x, y, z = value
        trans_x = c2[0] +scaled_points[key][0]
        trans_y = c2[1] + scaled_points[key][1]
        scaled_points2[key] = np.array([trans_x, trans_y, z])

    return scaled_points2, np.degrees(theta)  # 각도를 반환하여 확인 가능


def scale_z_values(transformed_positions, target_max_z=0.6):
    """
    z 값의 최댓값을 target_max_z로 스케일링.
    
    Args:
        transformed_positions (dict): 변환된 3D keypoints.
        target_max_z (float): 스케일링 후 z 값의 최댓값.

    Returns:
        dict: z 값이 스케일링된 변환된 3D keypoints.
    """
    # 모든 z 값을 추출
    z_values = np.array([value[2] for value in transformed_positions.values()])
    min_z, max_z = np.min(z_values), np.max(z_values)
    
    # 스케일 계산
    scale = target_max_z / max_z
    offset = -min_z * scale  # z 값을 양수로 유지하기 위해 최소값을 보정

    # z 값 스케일링 적용
    scaled_positions = {}
    for key, value in transformed_positions.items():
        x, y, z = value
        scaled_z = z * scale + offset
        scaled_positions[key] = np.array([x, y, scaled_z], dtype=np.float32)
    
    return scaled_positions


def rotate_around_z_180(scaled_positions):

    # Z축 기준 180도 회전 행렬
    rotation_matrix = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ])

    # 좌표 회전
    rotated_positions = {}
    for key, value in scaled_positions.items():
        rotated_xyz = np.dot(rotation_matrix, value)
        rotated_positions[key] = rotated_xyz

    return rotated_positions


def rotate_around_z_90(key_points_position, clockwise=True):

    rotated_positions = {}
    for key, value in key_points_position.items():
        x, y, z = value
        if clockwise:
            # 시계 방향 90도 회전
            rotated_x = y
            rotated_y = -x
        else:
            # 반시계 방향 90도 회전
            rotated_x = -y
            rotated_y = x
        rotated_positions[key] = np.array([rotated_x, rotated_y, z], dtype=np.float32)
    
    return rotated_positions


def calculate_intersections(Ran, p1, v1, v2):
    # Ran_x: Ran에서 v2 방향 직선이 v1과 만나는 점
    t_x = np.dot(Ran - p1, v2) / np.dot(v2, v2)
    Ran_x = p1 + t_x * v2
    
    # Ran_y: Ran에서 v1 방향 직선이 v2와 만나는 점
    t_y = np.dot(Ran - p1, v1) / np.dot(v1, v1)
    Ran_y = p1 + t_y * v1
    
    return Ran_x, Ran_y

def calculate_ground_truth(Ran_x, Ran_y, p1, p2, p3):
    gt_x = (Ran_x - p1) / (p2 - p1)
    gt_y = (Ran_y - p1) / (p3 - p1)
    return np.array([gt_x[0], gt_y[1]])
