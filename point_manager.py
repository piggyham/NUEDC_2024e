class Point:
    """单个点位数据存储"""
    def __init__(self, index, x, y, z):
        """
        :param index: 点位编号 (0-18)
        :param x: X坐标
        :param y: Y坐标
        :param z: Z坐标
        :param servo_corrections: 舵机修正量列表 [s1, s2, s3]
        """
        self.index = index
        self.coords = [float(x), float(y), float(z)]

class PointManager:
    """点位管理系统"""
    MAX_POINTS = 19
    
    def __init__(self):
        self.points = [None] * self.MAX_POINTS  # 预分配19个位置
        
    def add_point(self, index, x, y, z):
        """添加/更新点位"""
        if 0 <= index < self.MAX_POINTS:
            self.points[index] = Point(index, x, y, z)
        else:
            print(f"错误:点位编号需在0-{self.MAX_POINTS-1}之间")

    def get_point(self, index):
        """获取指定点位"""
        if 0 <= index < self.MAX_POINTS:
            return self.points[index]
        print(f"错误：无效的点位编号 {index}")
        return None

    def show_all(self):
        """显示所有点位信息"""
        for p in self.points:
            if p:
                print(f"点位{p.index}: ({p.coords[0]}, {p.coords[1]}, {p.coords[2]})")

#数据表
def point_information():
    pm = PointManager()
    # 添加格子数据
    pm.add_point(0,85, -40, -98)
    pm.add_point(1,80, -15, -90)
    pm.add_point(2,79, 9, -90)
    pm.add_point(3,75, 38, -95)
    pm.add_point(4,75, 65, -98)
    pm.add_point(5,-88.5, -42, -95)
    pm.add_point(6,-87, -14, -90)
    pm.add_point(7,-80, 14, -90)
    pm.add_point(8,-80, 35, -90)
    pm.add_point(9,-80, 63, -95)
    return pm
