


import cv2
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser




def parse_args():
    parser = ArgumentParser()
    parser.add_argument('data_path',type=str, help='Video path')
    args = parser.parse_args()
    return args


# 비디오에서 첫 번째 프레임을 추출하는 함수
def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # OpenCV의 BGR 이미지를 RGB로 변환 (matplotlib는 RGB 형식을 사용)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        print("Error: Could not read the video.")
        return None

# 마우스 이벤트 콜백 함수
def on_mouse_move(event):
    if event.xdata is not None and event.ydata is not None:
        # xdata, ydata는 소수점 값이므로 정수로 변환하여 출력
        print(f"Mouse coordinates: x={int(event.xdata)}, y={int(event.ydata)}")

args = parse_args()

#video_path = "C:/Users/admin/source/SST/samples/ablation/orange_view1_sideview.mp4"
video_path = args.data_path



# 비디오에서 첫 프레임을 추출
frame = extract_first_frame(video_path)

if frame is not None:
    # 이미지 출력
    fig, ax = plt.subplots()
    ax.imshow(frame)
    ax.scatter(1712, 510, color='red', s=1)
    ax.scatter(1739, 495, color='blue', s=1)
    # 마우스 이동 이벤트를 연결
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # 플롯 표시
    plt.show()