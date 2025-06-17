import math
import time
import serial
from time import sleep
import numpy as np
from point_manager import point_information
l0 = 1000
l1 = 1050
l2 = 880
l3 = 1500
pi = math.pi
time = 1500  # 默认运动时间
end_included_angle=10*np.pi/180
#angle_change_flag=0
l4=2150
#chess_num=5

'''
class servoControl:
    def __init__(self, pwm_pin , angle=0):
         # 初始化舵机控制，设置引脚和角度范围。
        self.steering = AngularServo(
        pin=pwm_pin,
        min_angle=0,
        max_angle=180,
        min_pulse_width=0.0005, # 0.5ms
        max_pulse_width=0.0025, # 2.5ms
        frame_width=0.02 # 50Hz (1/50=0.02s)
        ) 
        self.steering.angle = angle
        sleep(0.2)
        self.steering.angle = None # 停止控制，信号线PWM输出停止

    def set_angle(self, angle):
        self.steering.angle = angle
        sleep(0.2)
#        self.steering.angle = None # 停止控制，信号线PWM输出停止

#定义气泵输出引脚 记得该引脚
pump_inhale=servoControl(12)
pump_deflate=servoControl(13)
'''

class Kinematics:
    # 机械臂关节参数（单位：毫米，放大10倍
    
    def __init__(self, port='/dev/ttyS0', baudrate=115200):
        """
        初始化串口
        :param port: 树莓派串口设备(默认为GPIO UART)
        :param baudrate: 波特率
        """
        try:
            self.uart = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
        except serial.SerialException as e:
            print(f"串口初始化失败: {e}")
            self.uart = None

        self.chess_num=5
        self.angle_change_flag=0


    def send_str(self, cmd: str):
        """发送字符串指令"""
        if self.uart and self.uart.is_open:
            self.uart.write(cmd.encode('utf-8'))
        else:
            print("串口未连接")
    
    def trans_angle2streer(self,theta1,theta2):
        streer1=np.pi/2-theta1
        streer2=theta1+theta2
        return streer1,streer2
    
    def solve_angles(self,x,y,z):
        #if abs(x)>100 or y<0 or y>200 or z>100 or z<-100:
         #   print("WARNING--------输入越界------------WARNING")
          #  return 45.0,8.27,94.48,72.23
        if self.angle_change_flag==1:
            end_included_angle=15*np.pi/180
        else:
            end_included_angle = 10 * np.pi / 180
        x=x*10
        y=y*10
        z=z*10
        yaw=np.arctan(x/y)
        streer0=yaw
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
            ])
        x_body,y_body,z_body=tuple(Rz@np.array([x, y, z]))
        #待测试点1
        # print(x_body,y_body,z_body)
        z_body=z_body+l3*np.cos(end_included_angle)
        y_body=y_body-l3*np.sin(end_included_angle)
        theta1_1=np.arctan((z_body)/(y_body))+np.arccos((np.square(y_body)+np.square(z_body)+(l1*l1)-(l2*l2))/(2*l1*np.sqrt(np.square(y_body) + np.square(z_body))))
        if 0<theta1_1<np.pi:
            theta2_1=np.arctan((l1*np.sin(theta1_1)-(z_body))/((y_body)-(l1*np.cos(theta1_1))))
            (streer1,streer2)=self.trans_angle2streer(theta1_1,theta2_1)
            streer3=np.pi-streer1-streer2-end_included_angle
            # print(np.degrees(streer0),np.degrees(streer1),np.degrees(streer2),np.degrees(streer3))
            return np.degrees(streer0),np.degrees(streer1),np.degrees(streer2),np.degrees(streer3)
        theta1_2=np.arctan((z_body)/(y_body))-np.arccos((np.square(y_body)+np.square(z_body)+(l1*l1)-(l2*l2))/(2*l1*np.sqrt(np.square(y_body) + np.square(z_body))))
        theta2_2=np.arctan((l1*np.sin(theta1_2)-(z_body))/((y_body)-(l1*np.cos(theta1_2))))
        (streer1,streer2)=self.trans_angle2streer(theta1_2,theta2_2)
        streer3=np.pi-streer1-streer2-end_included_angle
        # print(np.degrees(streer0), np.degrees(streer1), np.degrees(streer2), np.degrees(streer3))
        return np.degrees(streer0),np.degrees(streer1),np.degrees(streer2),np.degrees(streer3)
    
    def steering_move1(self,x:float,y:float,z:float,time1:int):
        self.time=time1
        servo_angle0,servo_angle1,servo_angle2,servo_angle3=self.solve_angles(x,y,z)
        servo_pwm0 = int(1530-2000.0 * servo_angle0 / 270.0)
        servo_pwm1 = int(1500+2000.0 * servo_angle1 / 270.0)
        servo_pwm2 = int(1500+2000.0 * servo_angle2 / 270.0)
        servo_pwm3 = int(1500+2000.0 * servo_angle3 / 270.0)
        # print(servo_pwm0,servo_pwm1,servo_pwm2,servo_pwm3)
        #根据机械臂ID号输出指令
        arm_str = ("{{#000P{0:04d}T{4:04d}!#001P{1:04d}T{4:04d}!#002P{2:04d}T{4:04d}!#003P{3:04d}T{4:04d}!}}".format(servo_pwm0,servo_pwm1,servo_pwm2,servo_pwm3,self.time))
        self.send_str(arm_str)
        sleep(1)
        return 0

    def steering_move(self,x:float,y:float,z:float,time1:int = 1000):
        x = -x
        y = 145-y
        self.steering_move1(x,y,z,time1)

    def pump_inhale(self):
        self.send_str("{#004P2500T1000!}")
        sleep(2)
        self.send_str("{#004P0500T1000!}")
        sleep(1)
        # print("12")

    def pump_deflate_on(self):
        self.send_str("{#005P2500T1000!}")
        sleep(1)
    def pump_deflate_off(self):#停止放弃
        self.send_str("{#005P0500T1000!}")
        sleep(0.1)
    def arm_init(self):
        self.steering_move(-80,80,-10)
        
    def arm_middle(self):
        self.steering_move(x=0, y=30.0, z=-50.0, time1=1000)

    def arm_move_get_TARCHE(self, mode=0):
        pm=point_information()
        if mode==0:
            initial_chess = pm.get_point(5 - self.chess_num)
        else:
            initial_chess = pm.get_point(10 - self.chess_num)

        if self.chess_num==5:
            self.angle_change_flag=1
        self.arm_middle()
        #self.steering_move(initial_chess.coords[0], initial_chess.coords[1], -50, time1=1000)
        self.steering_move(initial_chess.coords[0], initial_chess.coords[1], initial_chess.coords[2], time1=1000)
        self.pump_inhale()
        self.steering_move(initial_chess.coords[0], initial_chess.coords[1], -50, time1=1000)
        #self.arm_middle()
        self.angle_change_flag = 0
        self.chess_num=self.chess_num-1

    def arm_move_callback(self,x,y,z):
        self.steering_move(x,y,z, time1=1000)

    def __del__(self):
        if self.uart and self.uart.is_open:
            self.uart.close()
            

if __name__ == "__main__":
    # 示例用法
    #arm = Kinematics(port='/dev/ttyUSB0')  # 根据实际串口设备修改

    arm = Kinematics(port='COM14')
    arm.pump_deflate_off()
    arm.arm_move_get_TARCHE(mode=0)
