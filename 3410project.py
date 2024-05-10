# 导入必要的库
import cv2
import mediapipe as mp
import time
import random

# 初始化MediaPipe Hands模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# 定义蛇类
class Snake:
    def __init__(self):
        self.score = 0
        self.points = []  # 蛇身节点列表
        self.allowedLength = 100  # 蛇身固定长度
        self.currentLength = 0

        self.foodPoint = (0, 0)  # 食物的起始位置
        self.randomFoodLocation()  # 随机改变食物的位置

    def randomFoodLocation(self):
        # x在100至300之间，y在100至300之间，随机取一个整数
        self.foodPoint = (
            random.randint(100, 300),  # 限制 x 范围在 100 到 300 之间
            random.randint(100, 300)   # 限制 y 范围在 100 到 300 之间
        )

    def update_snake(self, index_finger_tip):
        # 更新蛇身节点列表
        self.points.append(index_finger_tip)
        self.currentLength = sum(self.calculate_distance(self.points[i], self.points[i + 1]) for i in range(len(self.points) - 1))

        # 调整蛇身长度
        while self.currentLength > self.allowedLength and len(self.points) > 1:
            self.points.pop(0)
            self.currentLength = sum(self.calculate_distance(self.points[i], self.points[i + 1]) for i in range(len(self.points) - 1))

    def calculate_distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# 创建蛇类实例
snake = Snake()

# 图像处理函数
def get_image_brightness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = cv2.mean(gray_image)[0]
    return brightness

def adjust_brightness(image, factor):
    adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted_image

def process_frame(img):
    start_time = time.time()
    h, w = img.shape[0], img.shape[1]
    img = cv2.flip(img, 1)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    brightness = get_image_brightness(img)
    print(f'当前亮度: {brightness}')

    # 根据预定义的阈值调整图像亮度
    if brightness < threshold_darkness:
        img = adjust_brightness(img, brightness_factor_dark)
    elif brightness > threshold_brightness:
        img = adjust_brightness(img, brightness_factor_bright)

    # 使用MediaPipe处理手部信息
    results = hands.process(img_RGB)
    if results.multi_hand_landmarks:
        handness_str = ''
        index_finger_tip_str = ''
        
        # 循环遍历每个检测到的手
        for hand_idx in range(len(results.multi_hand_landmarks)):
            hand_21 = results.multi_hand_landmarks[hand_idx]
            mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS)
            temp_handness = results.multi_handedness[hand_idx].classification[0].label
            handness_str += '{}:{} '.format(hand_idx, temp_handness)
            cz0 = hand_21.landmark[0].z

            # 循环遍历手部的每个关键点
            for i in range(21):
                cx = int(hand_21.landmark[i].x * w)
                cy = int(hand_21.landmark[i].y * h)
                cz = hand_21.landmark[i].z
                depth_z = cz0 - cz
                radius = max(int(6 * (1 + depth_z * 5)), 0)

                if i == 8:  # 强调食指指尖所在位置
                    img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
                    index_finger_tip_str += '{}:{:.2f} '.format(hand_idx, depth_z)
                    snake.update_snake((cx, cy))  # 更新蛇身

        # 在图像上绘制蛇身
        for i in range(len(snake.points) - 1):
            cv2.line(img, snake.points[i], snake.points[i + 1], (0, 255, 0), 7)

        # 判断是否吃到食物
        if (
            snake.foodPoint[0] < snake.points[-1][0] < snake.foodPoint[0] + 20
            and snake.foodPoint[1] < snake.points[-1][1] < snake.foodPoint[1] + 20
        ):
            snake.allowedLength += 50  # 增加蛇身的固定长度
            snake.score += 1  # 得分加一
            snake.randomFoodLocation()  # 随机改变食物的位置

        # 在图像上绘制食物
        cv2.rectangle(img, snake.foodPoint, (snake.foodPoint[0] + 20, snake.foodPoint[1] + 20), (0, 0, 255), -1)

        scaler = 1
        img = cv2.putText(img, handness_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        img = cv2.putText(img, index_finger_tip_str, (25 * scaler, 150 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        img = cv2.putText(img, '得分: ' + str(snake.score), (25 * scaler, 200 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)

        # 判断是否达到胜利条件
        if snake.score >= 15:
            img = cv2.putText(img, '胜利!', (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
            img = cv2.putText(img, '按 "R" 重新开始', (w // 2 - 100, h // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        end_time = time.time()
        FPS = 1 / (end_time - start_time)
        scaler = 1
        img = cv2.putText(img, 'FPS  ' + str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)

    return img

# 调整亮度的阈值
threshold_brightness = 100
threshold_darkness = 50
brightness_factor_dark = 1.5  # 根据实际情况调整的调整因子
brightness_factor_bright = 0.7  # 根据实际情况调整的调整因子

# 打开摄像头
cap = cv2.VideoCapture(1)
cap.open(0)

# 主循环用于捕获和处理帧
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = process_frame(frame)
    cv2.imshow('my_window', frame)

    key = cv2.waitKey(1)
    if key in [ord('q'), 27]:
        break
    elif key == ord('r') and snake.score >= 15:
        snake = Snake()  # 重新开始游戏

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
