import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QWidget, QLabel, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import pickle
import chess_decide
import os  # 添加这行导入
from plusarm_control import Kinematics
from time import sleep

K_fix_x = 0.83
K_fix_y = 0.83
k_rate = 0.7
camera_num = 0

BLACK = 1
WHITE = 0
BLANK = -1
infinity = 100000

UP = 654
DOWN = 655


class target_class():
    target = [-3,3]
    def get_target(self):
        x = self.target[0]
        y = self.target[1]
        return [x,y]
    def change_target(self,target1):
        print(f'change target to{target1}')
        self.target = target1



class ChessGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_state()

        self.lower_blue = np.array([100, 150, 50])  # 较低的HSV阈值(蓝色)
        self.upper_blue = np.array([140, 255, 255])  # 较高的HSV阈值(蓝色)


    def init_state(self):
        """初始化所有状态变量"""
        self.M = None
        self.CALIBRATED = False
        self.show_processed = False
        self.last_warped = None
        self.last_processed = None
        self.is_load = False
        self.arm = Kinematics(port='COM14')
        self.arm.pump_deflate_off()
        self.target = target_class()
        self.pump_position = None
        self.pump_input =[0,0]
        self.old_cell  = np.full((3, 3), BLANK, dtype=int)


        self.grid_centers = np.full((3, 3), None)

        # 重新初始化摄像头
        if hasattr(self, 'cap'):
            self.cap.release()
        self.cap = cv2.VideoCapture(camera_num)
        if not self.cap.isOpened():
            print("错误：无法访问摄像头！")
            sys.exit()

        # 重置状态显示
        self.status_label.setText("状态: 等待矫正...")

        self.pd_timer_started = False  # 重置定时器状态

        self.pd_timer = QTimer(self)  # 创建定时器
        self.pd_timer.timeout.connect(self.execute_pd)  # 连接信号到槽
        self.pd_timer_started = False  # 跟踪定时器状态

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("三子棋识别系统")
        self.setGeometry(100, 100, 900, 700)

        # 主界面布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        # 左侧控制面板
        self.control_panel = QGroupBox("控制面板")
        self.control_layout = QVBoxLayout()

        # 按钮区域
        self.btn_load = QPushButton("加载矫正矩阵 (Load)")
        self.btn_save = QPushButton("保存矫正矩阵 (Save)")
        self.btn_detect = QPushButton("棋盘矫正 (Detect)")
        self.btn_action = QPushButton("执行动作 (Action)")
        self.btn_reset = QPushButton("重置系统 (Reset)")  # 新增重置按钮

        # 按钮样式
        button_style = """
            QPushButton {
                padding: 15px;
                font-size: 24px;
                min-width: 250px;
                min-height: 80px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        reset_button_style = """
            QPushButton {
                padding: 15px;
                font-size: 24px;
                min-width: 250px;
                min-height: 80px;
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """
        self.btn_load.setStyleSheet(button_style)
        self.btn_save.setStyleSheet(button_style)
        self.btn_detect.setStyleSheet(button_style)
        self.btn_action.setStyleSheet(button_style)
        self.btn_reset.setStyleSheet(reset_button_style)  # 设置重置按钮样式

        # 状态显示
        self.status_label = QLabel("状态: 等待矫正...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 28px; color: #333; padding: 10px;")

        # 添加到控制面板
        self.control_layout.addWidget(self.btn_action)
        self.control_layout.addWidget(self.btn_detect)
        self.control_layout.addWidget(self.btn_load)
        self.control_layout.addWidget(self.btn_save)
        self.control_layout.addWidget(self.btn_reset)  # 添加重置按钮
        self.control_layout.addWidget(self.status_label)
        self.control_layout.addStretch()
        self.control_panel.setLayout(self.control_layout)

        # 右侧图像显示区域
        self.image_panel = QGroupBox("图像+显示")
        self.image_layout = QVBoxLayout()

        # 原始图像显示
        self.original_label = QLabel("原始摄像头画面")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 1px solid #ddd; min-height: 240px;")

        # 矫正后图像显示
        self.warped_label = QLabel("矫正后画面")
        self.warped_label.setAlignment(Qt.AlignCenter)
        self.warped_label.setStyleSheet("border: 1px solid #ddd; min-height: 300px;")

        # 添加到图像面板
        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.warped_label)
        self.image_panel.setLayout(self.image_layout)

        # 添加到主布局
        self.main_layout.addWidget(self.control_panel, 1)
        self.main_layout.addWidget(self.image_panel, 1)

        # 连接按钮信号
        self.btn_load.clicked.connect(self.load_M)
        self.btn_save.clicked.connect(self.save_M)
        self.btn_detect.clicked.connect(self.detect_board)
        self.btn_action.clicked.connect(self.perform_action)
        self.btn_reset.clicked.connect(self.reset_system)  # 连接重置按钮

        # 定时器更新摄像头画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms更新一次

    def execute_pd(self):
        """每秒执行一次 pd 函数"""
        self.pd()

    def start_pd_timer(self):
        """启动 pd 定时器"""
        if not self.pd_timer_started:
            self.pd_timer.start(1100)  # 1000毫秒 = 1秒
            self.pd_timer_started = True
            self.status_label.setText("状态: pd函数运行中")

    def stop_pd_timer(self):
        """停止 pd 定时器"""
        if self.pd_timer_started:
            self.pd_timer.stop()
            self.pd_timer_started = False

    def original_point_to_physical(self, point_orig):
        """
        将原始图像中的点（假设在棋盘平面上）转换到棋盘物理坐标系（单位cm）
        参数:
            point_orig: 原始图像中的点坐标 (x, y)
        返回:
            (x_cm, y_cm): 物理坐标
        """
        # 将原始图像点转换为齐次坐标
        x, y = point_orig
        point_homogeneous = np.array([x, y, 1], dtype=np.float32)

        # 使用透视变换矩阵M将点变换到矫正图像坐标
        transformed = np.dot(self.M, point_homogeneous)
        # 防止除以0
        if transformed[2] < 1e-10:
            return (0, 0)
        x_warped = transformed[0] / transformed[2]
        y_warped = transformed[1] / transformed[2]

        # 将矫正图像坐标转换为物理坐标
        # 棋盘实际尺寸9cm，矫正图像大小300x300，所以每像素0.03cm
        x_cm = (x_warped - 150) * 0.03
        y_cm = (150 - y_warped) * 0.03  # 注意y轴方向：矫正图像中y向下，物理坐标系y向上

        return (x_cm*K_fix_x, y_cm*K_fix_y)

    def reset_system(self):
        """重置系统到初始状态"""
        self.arm_relax()
        self.init_state()
        # 清空图像显示
        self.original_label.clear()
        self.original_label.setText("原始摄像头画面")
        self.warped_label.clear()
        self.warped_label.setText("矫正后画面")
        self.stop_pd_timer()  # 重置时停止定时器
        print("系统已重置")

    def pixel_to_real(self, x_pixel, y_pixel, M_inv):
        """
        将像素坐标转换为实际物理坐标(cm)

        参数:
            x_pixel, y_pixel: 矫正后图像中的像素坐标
            M_inv: 透视变换的逆矩阵

        返回:
            (x_cm, y_cm): 物理坐标系中的坐标(cm)
        """
        # 1. 将矫正后图像中的点转换回原始图像坐标
        point = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
        orig_point = cv2.perspectiveTransform(point, M_inv)[0][0]

        # 2. 计算棋盘中心在原始图像中的位置
        center_point = np.array([[[150, 150]]], dtype=np.float32)
        orig_center = cv2.perspectiveTransform(center_point, M_inv)[0][0]

        # 3. 计算物理坐标
        # 棋盘实际尺寸
        board_size_cm = 9.0

        # 计算点到中心的向量（像素单位）
        dx_pixel = orig_point[0] - orig_center[0]
        dy_pixel = orig_point[1] - orig_center[1]

        # 计算棋盘在原始图像中的尺寸（像素）
        # 使用棋盘四个角点计算平均尺寸
        corners = np.array([[[0, 0], [300, 0], [300, 300], [0, 300]]], dtype=np.float32)
        orig_corners = cv2.perspectiveTransform(corners, M_inv)[0]

        # 计算水平和垂直方向的平均尺寸
        width_pixel = (np.linalg.norm(orig_corners[1] - orig_corners[0]) +
                       np.linalg.norm(orig_corners[2] - orig_corners[3])) / 2
        height_pixel = (np.linalg.norm(orig_corners[3] - orig_corners[0]) +
                        np.linalg.norm(orig_corners[2] - orig_corners[1])) / 2

        # 计算像素到厘米的转换比例
        scale_x = board_size_cm / width_pixel
        scale_y = board_size_cm / height_pixel

        # 转换为物理坐标（厘米）
        x_cm = dx_pixel * scale_x
        y_cm = -dy_pixel * scale_y  # 注意y轴方向反转

        return [x_cm, y_cm]

    # 其余方法保持不变...
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 显示原始图像
            original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = original_rgb.shape
            bytes_per_line = ch * w
            original_qimg = QImage(original_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.original_label.setPixmap(QPixmap.fromImage(original_qimg).scaled(
                320, 240, Qt.KeepAspectRatio))

            # 如果已矫正，显示矫正后图像或处理后的图像
            if self.CALIBRATED and self.M is not None:
                if not self.show_processed:
                    # 显示原始矫正图像
                    warped = cv2.warpPerspective(frame, self.M, (300, 300))
                    self.last_warped = warped.copy()
                    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                    h, w, ch = warped_rgb.shape
                    bytes_per_line = ch * w
                    warped_qimg = QImage(warped_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.warped_label.setPixmap(QPixmap.fromImage(warped_qimg).scaled(
                        300, 300, Qt.KeepAspectRatio))
                else:
                    # 显示处理后的图像
                    if self.last_processed is not None:
                        processed_rgb = cv2.cvtColor(self.last_processed, cv2.COLOR_BGR2RGB)
                        h, w, ch = processed_rgb.shape
                        bytes_per_line = ch * w
                        processed_qimg = QImage(processed_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        self.warped_label.setPixmap(QPixmap.fromImage(processed_qimg).scaled(
                            300, 300, Qt.KeepAspectRatio))

    def load_M(self):
        self.arm_relax()
        try:
            file_path = os.path.expanduser("D:/DeskTop/project/pythoh_project/NUEDC_2024_E/perspective_matrix.pkl")
            with open(file_path, 'rb') as f:
                self.M = pickle.load(f)
                self.CALIBRATED = True
                self.show_processed = False
                self.status_label.setText("状态: 已加载矫正矩阵")
                self.is_load = True
                self.M_inv = np.linalg.inv(self.M)
                print("成功加载保存的透视变换矩阵！")
        except (FileNotFoundError, pickle.PickleError):
            self.status_label.setText("状态: 加载矩阵失败")
            print("未找到保存的矩阵文件或加载失败，需要重新校准")

    def save_M(self):
        self.arm_relax()
        if self.M is not None:
            try:
                file_path = os.path.expanduser("D:/DeskTop/project/pythoh_project/NUEDC_2024_E/perspective_matrix.pkl")
                with open(file_path, 'wb') as f:
                    pickle.dump(self.M, f)
                self.status_label.setText("状态: 矩阵已保存")
                print("透视变换矩阵已保存到 perspective_matrix.pkl 文件")
            except Exception as e:
                self.status_label.setText("状态: 保存矩阵失败")
                print(f"保存矩阵失败: {e}")
        else:
            self.status_label.setText("状态: 无矩阵可保存")
            print("没有可保存的矩阵，请先进行棋盘矫正")

    def detect_board(self):
        ret, frame = self.cap.read()
        if ret:
            if not self.is_load:
                self.M, self.CALIBRATED = self.rectangle_correction(frame)
                if self.CALIBRATED:
                    self.M_inv = np.linalg.inv(self.M)
            self.show_processed = False  # 重置为显示矫正图像
            self.detect_pump(frame)
            if self.CALIBRATED:
                self.status_label.setText("状态: 棋盘矫正完成")
            else:
                self.status_label.setText("状态: 棋盘矫正失败")

    def detect_pump(self, frame):
        """
        检测泵在原始图像中的位置，并转换为棋盘物理坐标系
        坐标系说明: 棋盘中心为原点(0,0), 水平向右为x正方向, 垂直向上为y正方向
        """
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建蓝色掩膜
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找到最大轮廓
            max_contour = max(contours, key=cv2.contourArea)

            # 计算轮廓中心
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 将原始图像中的点转换为物理坐标
                physical_pos = self.original_point_to_physical((cx, cy))

                # 存储泵的位置
                self.pump_position = physical_pos

                # 在原始图像上绘制检测结果
                cv2.drawContours(frame, [max_contour], -1, (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Pump: ({physical_pos[0]:.2f}, {physical_pos[1]:.2f}) cm",
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                print(f"泵位置检测成功: 图像坐标({cx}, {cy}) -> 物理坐标({physical_pos[0]:.2f}, {physical_pos[1]:.2f}) cm")
                return True, physical_pos

        print("警告: 未检测到泵位置")
        self.pump_position = None
        return False

    def perform_action(self):
        if self.CALIBRATED and self.M is not None:
            ret, frame = self.cap.read()
            if ret:
                # 执行处理并显示处理后的图像
                self.show_processed = True
                board_color = self.chess_detect(frame, self.M)
                board = chess_decide.ChessBoard()
                a = [board_color[0, 0], board_color[0, 1], board_color[0, 2], board_color[1, 0], board_color[1, 1],
                         board_color[1, 2], board_color[2, 0], board_color[2, 1], board_color[2, 2]]
                if board.loading(a):
                    cheat, move_from, move_to = self.check_cheat(board.cell)
                    if cheat:
                        x0 = move_to[0]
                        y0 = move_to[1]
                        x1 = move_from[0]
                        y1 = move_from[1]
                        chess_xy = self.chess_detect_xy(frame, self.M)
                        # self.detect_pump(frame)
                        # sleep(0.5)
                        # self.detect_pump(frame)
                        destination = chess_xy[x0][y0]
                        destination = self.grid_centers[x0][y0]
                        self.target.change_target(destination)
                        self.pump_input = self.target.get_target()
                        self.pump_move(self.pump_input)
                        self.pump_move(self.pump_input, DOWN)
                        self.arm.pump_inhale()
                        self.pump_move(self.pump_input)
                        destination = self.grid_centers[x1][y1]
                        self.target.change_target(destination)
                        self.pump_input = self.target.get_target()
                        self.pump_move(self.pump_input)
                        self.pump_move(self.pump_input, DOWN)
                        self.arm.pump_deflate_on()
                        self.pump_move(self.pump_input)
                        self.arm_relax()
                        self.arm.pump_deflate_off()

                    else:
                        down_piece = chess_decide.action(board)
                        for x in range(3):
                            for y in range(3):
                                self.old_cell[x][y] = board.cell[x][y]
                        destination = self.grid_centers[down_piece[0]][down_piece[1]]
                        self.target.change_target(destination)
                        self.arm.arm_move_get_TARCHE()
                        self.pump_input= self.target.get_target()
                        self.pump_move(self.pump_input)
                        self.pump_move(self.pump_input, DOWN)
                        self.arm.pump_deflate_on()
                        self.pump_move(self.pump_input)
                        self.arm_relax()
                        self.arm.pump_deflate_off()
                        # self.start_pd_timer()
                        self.detect_pump(frame)

                self.status_label.setText("状态: 动作执行完成")
        else:
            self.status_label.setText("状态: 未完成矫正")
            print('未完成矫正')

    def rectangle_correction(self, frame):
        print('rectangle_correction')
        M = None
        CALIBRATED = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 250)

        # 查找最大轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

            if len(approx) == 4:
                # 计算透视变换矩阵
                src_pts = self.order_points(
                    np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]], dtype="float32"))
                dst_pts = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                CALIBRATED = True
                print("棋盘矫正完成！")
                return M, CALIBRATED
            else:
                print("警告：未检测到四边形轮廓，请调整棋盘位置！")
                return M, CALIBRATED
        else:
            print("警告：未检测到棋盘，请确保棋盘在画面中！")
            return M, CALIBRATED

    def order_points(self, pts):
        # 将四个顶点排序为：左上、右上、右下、左下
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上（x+y最小）
        rect[2] = pts[np.argmax(s)]  # 右下（x+y最大）

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上（x-y最小）
        rect[3] = pts[np.argmax(diff)]  # 左下（x-y最大）
        return rect

    def chess_detect(self, frame, M):
        print('enter chess_detect')
        warped = cv2.warpPerspective(frame, M, (300, 300))
        board_color = np.full((3, 3), -1)
        # 棋子检测流程
        circle_positions = np.full((3, 3), None)  # 存储棋子圆心实际坐标
        self.grid_centers = np.full((3, 3), None)
        M_inv = np.linalg.inv(M)

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        grid_size = 300 // 3

        # 检测每个格子
        for i in range(3):
            for j in range(3):
                x1, y1 = j * grid_size, i * grid_size
                x2, y2 = (j + 1) * grid_size, (i + 1) * grid_size

                grid_center_x = (x1 + x2) // 2
                grid_center_y = (y1 + y2) // 2

                # 转换格子中心为实际坐标
                self.grid_centers[i, j] = self.pixel_to_real(grid_center_x, grid_center_y, M_inv)

                grid = warped_gray[y1:y2, x1:x2]

                # 霍夫圆检测（优化参数）
                circles = cv2.HoughCircles(
                    grid,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=20,
                    param1=50,
                    param2=25,  # 降低阈值以提高检测灵敏度
                    minRadius=30,
                    maxRadius=50
                )

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        # 计算全局坐标
                        cx = x1 + circle[0]
                        cy = y1 + circle[1]
                        radius = circle[2]

                        circle_positions[i, j] = self.pixel_to_real(cx, cy, M_inv)

                        gray_roi = warped[cy - 5:cy + 5, cx - 5:cx + 5]
                        avg_saturation = np.mean(gray_roi[:, :, 1])  # 使用饱和度通道更可靠
                        # 分类逻辑
                        if avg_saturation < 125:  # 低灰度可能是黑色
                            color = (0, 0, 255)  # 红色标注白棋
                            board_color[i, j] = 1
                        else:  # 高饱和度可能是黑色
                            color = (0, 255, 0)  # 绿色标注白棋
                            board_color[i, j] = 0

                        # 绘制检测结果
                        cv2.circle(warped, (cx, cy), radius, color, 2)
                        cv2.rectangle(warped, (x1, y1), (x2, y2), (255, 0, 0), 1)

        print("\n===== 格子中心坐标 (物理坐标系) =====")
        print("坐标系说明: 棋盘中心为原点(0,0), 水平向右为x正方向, 垂直向上为y正方向")
        for i in range(3):
            for j in range(3):
                x, y = self.grid_centers[i, j]
                print(f"格子({i},{j}): ({x:.2f}cm, {y:.2f}cm)")

        print("\n===== 棋子圆心坐标 (物理坐标系) =====")
        for i in range(3):
            for j in range(3):
                if circle_positions[i, j] is not None:
                    x, y = circle_positions[i, j]
                    print(f"棋子({i},{j}): ({x:.2f}cm, {y:.2f}cm)")
        # 保存处理后的图像
        self.last_processed = warped.copy()
        return board_color

    def chess_detect_xy(self, frame, M):
        print('enter chess_detect')
        warped = cv2.warpPerspective(frame, M, (300, 300))
        board_color = np.full((3, 3), -1)
        # 棋子检测流程
        circle_positions = np.full((3, 3), None)  # 存储棋子圆心实际坐标
        self.grid_centers = np.full((3, 3), None)
        M_inv = np.linalg.inv(M)

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        grid_size = 300 // 3

        # 检测每个格子
        for i in range(3):
            for j in range(3):
                x1, y1 = j * grid_size, i * grid_size
                x2, y2 = (j + 1) * grid_size, (i + 1) * grid_size

                grid_center_x = (x1 + x2) // 2
                grid_center_y = (y1 + y2) // 2

                # 转换格子中心为实际坐标
                self.grid_centers[i, j] = self.pixel_to_real(grid_center_x, grid_center_y, M_inv)

                grid = warped_gray[y1:y2, x1:x2]

                # 霍夫圆检测（优化参数）
                circles = cv2.HoughCircles(
                    grid,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=20,
                    param1=50,
                    param2=25,  # 降低阈值以提高检测灵敏度
                    minRadius=30,
                    maxRadius=50
                )

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        # 计算全局坐标
                        cx = x1 + circle[0]
                        cy = y1 + circle[1]
                        radius = circle[2]

                        circle_positions[i, j] = self.pixel_to_real(cx, cy, M_inv)

                        gray_roi = warped[cy - 5:cy + 5, cx - 5:cx + 5]
                        avg_saturation = np.mean(gray_roi[:, :, 1])  # 使用饱和度通道更可靠
                        # 分类逻辑
                        if avg_saturation < 125:  # 低灰度可能是黑色
                            color = (0, 0, 255)  # 红色标注白棋
                            board_color[i, j] = 1
                        else:  # 高饱和度可能是黑色
                            color = (0, 255, 0)  # 绿色标注白棋
                            board_color[i, j] = 0

                        # 绘制检测结果
                        cv2.circle(warped, (cx, cy), radius, color, 2)
                        cv2.rectangle(warped, (x1, y1), (x2, y2), (255, 0, 0), 1)

        print("\n===== 格子中心坐标 (物理坐标系) =====")
        print("坐标系说明: 棋盘中心为原点(0,0), 水平向右为x正方向, 垂直向上为y正方向")
        for i in range(3):
            for j in range(3):
                x, y = self.grid_centers[i, j]
                print(f"格子({i},{j}): ({x:.2f}cm, {y:.2f}cm)")

        print("\n===== 棋子圆心坐标 (物理坐标系) =====")
        for i in range(3):
            for j in range(3):
                if circle_positions[i, j] is not None:
                    x, y = circle_positions[i, j]
                    print(f"棋子({i},{j}): ({x:.2f}cm, {y:.2f}cm)")
        # 保存处理后的图像
        self.last_processed = warped.copy()
        return circle_positions


    def pump_move(self,destination,mode = UP):
        x = destination[0]*10
        y = destination[1]*10
        if mode == UP:
            self.arm.steering_move(x,y,-50)
        elif mode == DOWN:
            self.arm.steering_move(x,y,-90)
        else :
            print("ERROR---------错误mode----------ERROR")

    def check_cheat(self,newcell):
        change_color = -1
        cheat = False
        move_from = [-1, -1]
        move_to = [-1, -1]
        for x in range(3):
            for y in range(3):
                if self.old_cell[x][y] !=-1:
                    if newcell[x][y] != self.old_cell[x][y]:
                        change_color = self.old_cell[x][y]
                        move_from = [x,y]
                        cheat = True
                        break
            if cheat:
                    break
        if cheat == False:
            return cheat, move_from, move_to
        for x in range(3):
            for y in range(3):
                if newcell[x][y] == change_color and self.old_cell[x][y] == -1:
                    move_to =[x,y]
        print(f'你放屁！！！！你把{move_from}挪到了{move_to}!!!!!!')
        return cheat,move_from,move_to


    def arm_relax(self):
        self.arm.steering_move(-80,80,-10)

    def pd(self):
        ret, frame = self.cap.read()
        if ret:
            if self.detect_pump(frame):
                # print(f'pump_position = {self.pump_position}')
                distance = abs(self.target.get_target()[0]-self.pump_position[0])+abs(self.target.get_target()[1]-self.pump_position[1])
                print(f'distance={distance}')
                if distance < 0.2:
                    self.stop_pd_timer()
                    self.pump_move(self.pump_input,DOWN)
                    self.arm.pump_deflate_on()
                    self.pump_move(self.pump_input)
                    self.arm_relax()
                    self.arm.pump_deflate_off()

                else:
                    error_x = self.target.get_target()[0] - self.pump_position[0]
                    error_y = self.target.get_target()[1] - self.pump_position[1]
                    self.pump_input[0] = self.pump_input[0] +error_x*k_rate
                    self.pump_input[1] = self.pump_input[1] + error_y*k_rate
                    print(f'pump_input = {self.pump_input},target = {self.target.get_target()},pump_position = {self.pump_position}')
                    self.pump_move(destination = self.pump_input)


    def closeEvent(self, event):
        # 关闭时释放资源
        self.arm_relax()
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChessGUI()
    window.show()
    sys.exit(app.exec_())
