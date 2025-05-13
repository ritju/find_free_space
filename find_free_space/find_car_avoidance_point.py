import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer,GoalResponse,CancelResponse
from rclpy.qos import qos_profile_sensor_data, DurabilityPolicy,ReliabilityPolicy,QoSProfile,HistoryPolicy
from geometry_msgs.msg import PoseStamped, Quaternion
from capella_ros_msg.srv import IsCarPassable
import tf2_ros
import numpy as np
import time
import math
import cv2
from nav_msgs.msg import OccupancyGrid
from capella_ros_msg.action import FindCarAvoidancePoint
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import PolygonStamped


class CarAvoidancePointActionServer(Node):
    def __init__(self):
        super().__init__('find_free_space_action_server')
        self.get_logger().info('find_free_space_action_server started.')

        # 初始化参数
        self.robot_width = 1.0
        self.vehicle_width = 2.0
        self.redundancy_distance = 0.5
        self.search_interval = 0.5
        self.action_goal_handle_msg = None
        self.search_radius = 4.0

        self.init_params()

        callback_gp1 = MutuallyExclusiveCallbackGroup()
        callback_gp2 = MutuallyExclusiveCallbackGroup()
        callback_gp3 = MutuallyExclusiveCallbackGroup()
        callback_gp4 = MutuallyExclusiveCallbackGroup()

        self.footprint_sub_ = self.create_subscription(
            PolygonStamped, 
            self.topic_name_footprint,
            self.footprint_sub_callback,
            1,
            callback_group=callback_gp4)

        self.robot_pose = PoseStamped()
        # 创建一个timer，用于实时获取机器人的位姿
        self.get_robot_pose_timer_ = self.create_timer(timer_period_sec=0.1, callback=self.get_robot_pose_timer_callback)

        self.polygons = []
        self.verties = []
        # 创建一个timer,用于实时得到距离机器人最近的通道位姿。
        # self.get_verties_timer = self.create_timer(timer_period_sec=0.5, callback=self.get_verties_callback)        
        
        # action
        action_server_feedback_qos = QoSProfile(depth=1)
        action_server_feedback_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.action_server = ActionServer(self,FindCarAvoidancePoint,'/find_car_avoidance_point_action',
                                        self.action_goal_callback,
                                        callback_group=callback_gp1,
                                        feedback_pub_qos_profile=action_server_feedback_qos)#
        # tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.global_costmap_sub = self.create_subscription(
            OccupancyGrid,
            self.topic_name_global_costmap,
            self.global_costmap_callback,
            10,
            callback_group=callback_gp2)
        self.global_costmap = None
        # 检查pose能否避让的服务
        self.check_avoidance_service = self.create_client(IsCarPassable, '/check_car_passable',callback_group=callback_gp3)

    def init_params(self):
        self.declare_parameter("topic_name_global_costmap", "")
        self.declare_parameter("service_name_check_car_passble", "")
        self.declare_parameter("topic_name_footprint", "")
        self.declare_parameter("redundancy_distance", 0.3)
        self.declare_parameter("search_interval", 0.3)
        self.declare_parameter("search_radius", 2.0)

        self.topic_name_global_costmap = self.get_parameter("topic_name_global_costmap").value
        self.service_name_check_car_passble = self.get_parameter("service_name_check_car_passble").value
        self.topic_name_footprint = self.get_parameter("topic_name_footprint").value
        self.redundancy_distance = self.get_parameter("redundancy_distance").value
        self.search_interval = self.get_parameter("search_interval").value
        self.search_radius = self.get_parameter("search_radius").value

        self.get_logger().info(f'topic_name_global_costmap: {self.topic_name_global_costmap}')
        self.get_logger().info(f'service_name_check_car_passble: {self.service_name_check_car_passble}')
        self.get_logger().info(f'topic_name_footprint: {self.topic_name_footprint}')
        self.get_logger().info(f'redundancy_distance: {self.redundancy_distance}')
        self.get_logger().info(f'search_interval: {self.search_interval}')
        self.get_logger().info(f'search_radius: {self.search_radius}')
    
    def footprint_sub_callback(self, msg):
        points = msg.polygon.points
        edges_distance = np.array([
            round(np.linalg.norm(
                np.array([points[i].x, points[i].y]) - 
                np.array([points[(i+1)%4].x, points[(i+1)%4].y])
            ), 2)
            for i in range(4)
        ])
        self.robot_width = np.min(edges_distance)
        self.get_logger().info(f"robot_width: {self.robot_width}")
        # 取消订阅
        self.destroy_subscription(self.footprint_sub_)

    def global_costmap_callback(self, msg):
        self.global_costmap = msg
    
    # 用于实时获取机器人的位姿
    def get_robot_pose_timer_callback(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.robot_pose.header.stamp = self.get_clock().now().to_msg()
            self.robot_pose.header.frame_id = 'map'
            self.robot_pose.pose.position.x = trans.transform.translation.x
            self.robot_pose.pose.position.y = trans.transform.translation.y
            self.robot_pose.pose.orientation = trans.transform.rotation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'{e}')
        
        self.get_logger().info(f'机器人当前位姿: [{self.robot_pose.pose.position.x}, {self.robot_pose.pose.position.y}]', throttle_duration_sec=2)

    def get_verties_callback(self):
        if len(self.polygons == 0):
            pass
        else:
            current_polygon_verties = []
            for polygon in self.polygons:
                current_polygon_verties = np.array([[point.x,point.y] for point in polygon.points])
                robot_is_in_area = self.is_point_inside_parallelogram(self.robot_pose.pose.position.x,self.robot_pose.pose.position.y,current_polygon_verties)
                if robot_is_in_area:
                    self.verties = current_polygon_verties
                    break   

    def action_goal_callback(self, goal_handle):
        self.get_logger().info('开始寻找避让点...')
        # self.get_logger().info(f'goal_handle.request..{goal_handle.request}')
        self.action_goal_handle_msg = goal_handle.request
        self.vehicle_width = self.action_goal_handle_msg.car_size.y
        self.polygons = self.action_goal_handle_msg.polygons
        
        # 获取清洁区域信息
        # 每次需要用到self.verties时，调用一下 get_verties_callback()
        self.get_logger().info('寻找当前通行区域...')
        self.verties.clear()
        self.get_verties_callback()

        if len(self.verties) == 0:
            self.get_logger().error('未找到用于寻找停靠点的通道')
            goal_handle.abort()
            return FindCarAvoidancePoint.Result()
        
        # 计算总通行宽度
        self.get_logger().info('计算当前通行区域宽度...')
        total_passage_width = self.calculate_total_passage_width(cleaning_area_vertices)

        # 判断是否需要在通行区域外寻找
        required_width = self.robot_width + self.vehicle_width + self.redundancy_distance
        if required_width > total_passage_width:
            search_outside = True
        else:
            search_outside = False

        # 寻找停靠点
        self.get_logger().info('寻找停靠点...')
        avoidance_point = self.find_avoidance_point(robot_pose, cleaning_area_vertices, search_outside)
        self.get_logger().info(f'avoidance_point:{avoidance_point}')
        if avoidance_point is not None:
            self.get_logger().info(f'成功找到避让点{avoidance_point}')
            goal_handle.succeed()
            # goal_handle.abort()
            result = FindCarAvoidancePoint.Result()
            result.pose = avoidance_point
            self.get_logger().info(f'成功找到避让点*****')
            return result
        else:
            self.get_logger().info(f'无法找到避让点')
            goal_handle.abort()
            return FindCarAvoidancePoint.Result()

    def calculate_total_passage_width(self, vertices):
        # 假设为长方形，长边为通行方向，短边为通道宽度
        # p1 = np.array([vertices[0].x, vertices[0].y])
        # p2 = np.array([vertices[1].x, vertices[1].y])
        # p3 = np.array([vertices[2].x, vertices[2].y])
        # p4 = np.array([vertices[3].x, vertices[3].y])

        # 计算每两点之间的长边
        side_length_list = [math.dist([vertices[i][0],vertices[i][1]],[vertices[(i + 1) % 4][0],vertices[(i + 1) % 4][1]]) for i in range(4)]
        distance = min(side_length_list)

        return distance

    # 寻找所有的避让点
    def find_avoidance_point(self, robot_pose, cleaning_area_vertices, search_outside):
        if self.global_costmap is None:
            self.get_logger().error('全局代价地图未收到')
            return None

        map_info = self.global_costmap.info
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y
        resolution = map_info.resolution
        width = map_info.width
        height = map_info.height
        
        map = np.array(self.global_costmap.data)
        map[map == -1] = 50
        map = (100 - map) / 100 * 255
        map[map != 0] = 255
        costmap = np.ascontiguousarray(map.reshape((height,width)).astype(np.uint8)[::-1])

        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y

        # 计算最近的边界
        self.get_logger().info('寻找最近的边界...')
        nearest_boundary = self.find_nearest_boundary(robot_pose, cleaning_area_vertices)
        y_ = (nearest_boundary[0][1] - nearest_boundary[1][1])
        x_ = (nearest_boundary[0][0] - nearest_boundary[1][0])
        if x_ == 0.0:
            k = math.pi /2
        else:
            k = y_ / x_
        p1,p2 = nearest_boundary
        k = np.arctan(k)
        self.get_logger().error(f'k: {k/math.pi*180}')
        # 计算四个方向
        directions_ = [self.convert_angle_to_ros2(k+x) for x in [0.0,math.pi]]
        self.get_logger().info(f'kdirections_: {directions_}')
        directions = []
        for angle in directions_:
            q = Quaternion()
            q.x, q.y, q.z, q.w = quaternion_from_euler(0, 0, angle)
            directions.append(q)
        # 计算最近的边界直线，ax+by+c=0
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0] * p1[1] - p1[0] * p2[1]
        # 计算汽车和机器人之间的角度
        car_pose= self.action_goal_handle_msg.car_pose.pose.position
        y_ = (car_pose.y - robot_y)
        x_ = (car_pose.x - robot_x)
        if x_ == 0.0:
            robot_car_k = math.pi /2
        else:
            robot_car_k = y_ / x_
            robot_car_k = np.arctan(robot_car_k)
        k_diff = self.angle_diff(k,robot_car_k)
        if k_diff < math.pi/2:
            k = self.add_angles(k,math.pi)
            
        if not search_outside:
            # 计算搜索范围
            robot_x1 = robot_x + math.cos(k) * self.search_radius
            robot_y1 = robot_y + math.sin(k) * self.search_radius
            robot_x2 = robot_x
            robot_y2 = robot_y
        else:
            robot_x_opposite,robot_y_opposite = self.find_symmetric_point([robot_x,robot_y],[a,b,c])
            # 计算搜索范围
            robot_x1 = robot_x_opposite + math.cos(k) * self.search_radius
            robot_y1 = robot_y_opposite + math.sin(k) * self.search_radius
            robot_x2 = robot_x_opposite
            robot_y2 = robot_y_opposite
        # x = -(c+by)/a
        if a == 0.0:
            xx1 = p1[1]
            xx2 = p2[1]
        else:
            xx1 = -(c + b*robot_y1)/a
            xx2 = -(c + b*robot_y2)/a
        
        # 在区域内搜索，往边界靠近
        # 四个点按照顺序排序
        find_vertices = self.sort_quadrilateral_vertices([(robot_x1,robot_y1),(xx1,robot_y1),(xx2,robot_y2),(robot_x2,robot_y2)])
        
        self.get_logger().info('生成所有停靠点...')
        boundary_points = self.select_points_in_parallelogram(find_vertices,0.5,0.0)
        # 判断每个点是否里障碍物太近
        # 首先将位姿转换到map的像素点
        if len(boundary_points) > 0:
            boundary_points = np.array(boundary_points)
            boundary_points_pixel = (boundary_points - np.array([origin_x,origin_y])) / resolution
            boundary_points_pixel[:,1] = height - boundary_points_pixel[:,1]
            boundary_points_pixel[:,0] = np.clip(boundary_points_pixel[:,0],0,width-1)
            boundary_points_pixel[:,1] = np.clip(boundary_points_pixel[:,1],0,height-1)
            self.get_logger().info(f'一共{len(boundary_points)}个避障...')
            # 判断目标点附近是否有障碍物
            self.get_logger().info('排除障碍物点...')
            is_obstacle_index = [True if self.check_point_is_free(costmap,(x,y),5) else False for x,y in boundary_points_pixel]
            boundary_points = boundary_points[is_obstacle_index]
            self.get_logger().info(f'排除障碍物点后还剩{len(boundary_points)}个避障...')
            if len(boundary_points) > 0:
                # 对所有点进行排序，从近到远
                # 计算所有点到机器人的距离
                point_dist = np.array([math.dist(p,[robot_x,robot_y]) for p in boundary_points])
                print(boundary_points)
                boundary_points = sorted(boundary_points,key=lambda x:point_dist[(boundary_points==x).all(1)])
                for point in boundary_points:
                    point_x = point[0]
                    point_y = point[1]
                    for direction in directions:
                        avoidance_pose = PoseStamped()
                        avoidance_pose.header.stamp = self.get_clock().now().to_msg()
                        avoidance_pose.header.frame_id = 'map'
                        avoidance_pose.pose.position.x = point_x
                        avoidance_pose.pose.position.y = point_y
                        avoidance_pose.pose.orientation = direction
                        
                        avoidance_pose_msg = IsCarPassable.Request()
                        avoidance_pose_msg.robot_pose = avoidance_pose
                        avoidance_pose_msg.car_pose = self.action_goal_handle_msg.car_pose
                        avoidance_pose_msg.size = self.action_goal_handle_msg.car_size
                        check_avoidance_result = self.check_avoidance(avoidance_pose_msg)
                        print('check_avoidance_result  ',check_avoidance_result)
                        if check_avoidance_result:
                            robot_point = np.array([robot_x,robot_y])
                            robot_point_pixel = (robot_point - np.array([origin_x,origin_y])) / resolution
                            robot_point_pixel[1] = height - robot_point_pixel[1]
                            robot_point_pixel[0] = np.clip(robot_point_pixel[0],0,width-1)
                            robot_point_pixel[1] = np.clip(robot_point_pixel[1],0,height-1)
                            # robot_x_p,robot_y_p = robot_point_pixel
                            robot_x_p = int(robot_point_pixel[0])
                            robot_y_p = int(robot_point_pixel[1])

                            point = np.array([point_x,point_y])
                            point_pixel = (point - np.array([origin_x,origin_y])) / resolution
                            point_pixel[1] = height - point_pixel[1]
                            point_pixel[0] = np.clip(point_pixel[0],0,width-1)
                            point_pixel[1] = np.clip(point_pixel[1],0,height-1)
                            # point_x_p,point_y_p = point_pixel
                            point_x_p = int(point_pixel[0])
                            point_y_p = int(point_pixel[1])
                            bresenham_point = self.bresenham(robot_x_p,robot_y_p,point_x_p,point_y_p,costmap)
                            bresenham_point_value = np.array([costmap[x[1],x[0]] for x in bresenham_point])
                            
                            if (bresenham_point_value > 250).all():
                                return avoidance_pose
                            
                
                return None
            else:
                return None
        
        return None

    # 寻找最近的边界
    def find_nearest_boundary(self, robot_pose, cleaning_area_vertices):
        robot_point = np.array([robot_pose.pose.position.x, robot_pose.pose.position.y])
        min_distance = float('inf')
        nearest_boundary = None
        for i in range(4):
            p1 = np.array([cleaning_area_vertices[i][0], cleaning_area_vertices[i][1]])
            p2 = np.array([cleaning_area_vertices[(i + 1) % 4][0], cleaning_area_vertices[(i + 1) % 4][1]])
            # ax+by+c=0
            a = p1[1] - p2[1]
            b = p2[0] - p1[0]
            c = p1[0] * p2[1] - p2[0] * p1[1]
            # point to line dist
            distance = abs(a*robot_point[0] + b*robot_point[1] + c) / math.sqrt(a**2 + b**2)
            if distance < min_distance:
                min_distance = distance
                nearest_boundary = (p1, p2)
        return nearest_boundary

    # 发送服务，判断避让点是否能够让车通过
    def check_avoidance(self, avoidance_msg):
        while not self.check_avoidance_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('服务未就绪，等待中...')
    
        # future = self.check_avoidance_service.call_async(avoidance_msg)
        # self.get_logger().info(f'判断是否能通过:{avoidance_msg.robot_pose.pose.position}')
        result = self.check_avoidance_service.call(avoidance_msg)
        # self.get_logger().info(f'result.is_car_passable:{result.is_car_passable}')
        return result.is_car_passable
        # rclpy.spin_until_future_complete(self, future)
        # if future.result().is_car_passable:
        #     self.get_logger().info('成功找到避让点。')
        #     return True
        # else:
        #     self.get_logger().info('该点无法避障。')
        #     return False

    # 判断机器人是否在某个边框内
    def is_point_inside_parallelogram(self, robot_x, robot_y, vertices):
        """
        判断机器人是否在清洁区域内
        :param robot_x: 机器人的 x 坐标
        :param robot_y: 机器人的 y 坐标
        :param vertices: 清洁区域的四个顶点坐标
        :return: True 表示在区域内，False 表示在区域外
        """
        inside = False
        j = len(vertices) - 1
        for i in range(len(vertices)):
            xi, yi = vertices[i][0], vertices[i][1]
            xj, yj = vertices[j][0], vertices[j][1]

            intersect = ((yi > robot_y) != (yj > robot_y)) and \
                        (xj - xi) * (robot_y - yi) - (yj - yi) * (robot_x - xi) # 使用向量法，避免分母为0的情况
                        # (robot_x < (xj - xi) * (robot_y - yi) / (yj - yi) + xi)
            if intersect:
                inside = not inside

            j = i
        return inside
    
    # 选择平行四边形区域内的点
    def select_points_in_parallelogram(self, vertices, interval, boundary_distance):
        """
        选择平行四边形区域内的点
        :param vertices: 平行四边形的四个顶点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        :param interval: 间隔距离
        :param boundary_distance: 到边界的距离
        :return: 平行四边形区域内的点列表
        """
        print('vertices: ',vertices)
        min_x = min([v[0] for v in vertices]) + boundary_distance
        max_x = max([v[0] for v in vertices]) - boundary_distance
        min_y = min([v[1] for v in vertices]) + boundary_distance
        max_y = max([v[1] for v in vertices]) - boundary_distance
        points = []
        for x in np.arange(min_x, max_x, interval):
            for y in np.arange(min_y, max_y, interval):
                if self.is_point_inside_parallelogram(x, y, vertices):
                    points.append([x, y])
        return points
    
    # 对四边形的四个顶点进行排序，按左上角、右上角、右下角、左下角的顺序返回
    def sort_quadrilateral_vertices(self, vertices):
        """
        对四边形的四个顶点进行排序，按左上角、右上角、右下角、左下角的顺序返回
        :param vertices: 四边形的四个顶点列表，每个顶点是一个二元组 (x, y)
        :return: 排序后的顶点列表
        """
        # 按 y 坐标从小到大排序，如果 y 坐标相同，则按 x 坐标从小到大排序
        sorted_vertices = sorted(vertices, key=lambda point: (point[1], point[0]))

        # 前两个点是上方的点，后两个点是下方的点
        top_points = sorted_vertices[:2]
        bottom_points = sorted_vertices[2:]

        # 对上方的点按 x 坐标从小到大排序，得到左上角和右上角的点
        top_left, top_right = sorted(top_points, key=lambda point: point[0])

        # 对下方的点按 x 坐标从小到大排序，得到左下角和右下角的点
        bottom_left, bottom_right = sorted(bottom_points, key=lambda point: point[0])

        return [top_left, top_right, bottom_right, bottom_left]
    
    # 将角度转换为机器人坐标系下的角度
    def convert_angle_to_ros2(self, angle, input_in_degrees=False):
        """
        将角度转换为 ROS 2 中常用的弧度范围 (-π, π]。
        :param angle: 输入的角度值。
        :param input_in_degrees: 若为 True，则输入角度以度为单位；若为 False，则以弧度为单位。默认为 False。
        :return: 转换到 (-π, π] 范围内的弧度值。
        """
        # 如果输入是度，先将其转换为弧度
        if input_in_degrees:
            angle = math.radians(angle)

        # 把角度转换到 (-π, π] 范围内
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi

        return angle

    # 寻找点关于一条直线的对称点
    def find_symmetric_point(self, a, line_coeffs):
        """
        计算点 a 关于给定直线的对称点 b
        :param a: 已知点，格式为 (x, y)
        :param line_coeffs: 直线方程 Ax + By + C = 0 的系数，格式为 (A, B, C)
        :return: 对称点 b 的坐标，格式为 (x, y)
        """
        A, B, C = line_coeffs
        x1, y1 = a

        # 计算对称点的公式
        denominator = A ** 2 + B ** 2
        x2 = x1 - 2 * A * (A * x1 + B * y1 + C) / denominator
        y2 = y1 - 2 * B * (A * x1 + B * y1 + C) / denominator

        return (x2, y2)
    
    # 检查点附近是否有障碍物
    def check_point_is_free(self, image, center, radius=20):
        """
        判断以指定像素点为中心、半径为 x 像素的圆形区域内所有像素值是否都大于 0
        :param image: 输入的单通道图像（灰度图）
        :param center: 中心像素点的坐标 (x, y)
        :param x: 圆形区域的半径
        :return: 如果圆形区域内所有像素值都大于 0 返回 True，否则返回 False
        """
        height, width = image.shape

        # 遍历圆形区域内的所有像素
        for y in range(max(0, int(center[1] - radius)), min(height, int(center[1] + radius + 1))):
            for x in range(max(0, int(center[0] - radius)), min(width, int(center[0] + radius + 1))):
                # 计算当前像素到中心像素的距离
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                if distance <= radius:
                    # 检查像素值是否大于 1
                    if image[y, x] < 200:
                        return False
        return True

    # 判断两个点之间的角度差
    def angle_diff(self, a, b, use_abs=True):
        """
        判断以指定像素点为中心、半径为 x 像素的圆形区域内所有像素值是否都大于 0
        :param a: 角度a（弧度）
        :param b: 角度b（弧度）
        :param use_abs: 是否返回绝对值
        :return: a和b之间的角度差（弧度）
        """
        error = a - b
        if error < -math.pi:
                error = error + 2*math.pi #小于-pi加2pi
        elif error >= math.pi:
                error = error - 2*math.pi #大于pi减2pi
        else:
                pass
        if use_abs:
                return abs(error)
        else:
                return error
        
    # 返回两个角度相加的结果
    def add_angles(self, current_angle, rotation_angle):
        """
        计算机器人当前角度和旋转角度相加的结果，并将结果限制在 [-pi, pi] 范围内
        :param current_angle: 机器人当前角度（弧度）
        :param rotation_angle: 旋转角度（弧度）
        :return: 相加后的角度，范围在 [-pi, pi] 之间
        """
        # 计算相加后的角度
        new_angle = current_angle + rotation_angle
        # 将结果限制在 [-pi, pi] 范围内
        while new_angle > math.pi:
            new_angle -= 2 * math.pi
        while new_angle < -math.pi:
            new_angle += 2 * math.pi
        return new_angle

    
    def bresenham(self, current_x, current_y, target_x, target_y, map_array):
        """
        提取两点连线上的所有像素点
        :param current_x: 当前点的 x 坐标
        :param current_y: 当前点的 y 坐标
        :param target_x: 目标点的 x 坐标
        :param target_y: 目标点的 y 坐标
        :param map_array: 地图数组
        :return: 两点连线上的所有像素点列表
        """
        pixels = []
        dx = abs(target_x - current_x)
        dy = abs(target_y - current_y)
        sx = 1 if current_x < target_x else -1
        sy = 1 if current_y < target_y else -1
        err = dx - dy

        x, y = current_x, current_y
        self.get_logger().info(f'x: {current_x}, y: {current_y}')
        self.get_logger().info(f't_x: {target_x}, t_y: {target_y}')
        self.get_logger().info(f'm_x: {len(map_array[0])}, y: {len(map_array)}')
        while True:
            # 检查点是否在地图范围内
            if 0 <= x < len(map_array[0]) and 0 <= y < len(map_array):
                pixels.append((x, y))
            if x == target_x and y == target_y:
                break
            e2 = 2 * err
            if e2 > -dy:
                err = err - dy
                x = x + sx
                # self.get_logger().info(f'x: {x}, sx: {sx}')
            if e2 < dx:
                err = err + dx
                y = y + sy
                # self.get_logger().info(f'y: {y}, sy: {sy}')

        return np.array(pixels)

def main(args=None):
    rclpy.init(args=args)
    executor_ = MultiThreadedExecutor()
    car_avoidance_action_server = CarAvoidancePointActionServer()
    rclpy.spin(car_avoidance_action_server,executor=executor_)
    car_avoidance_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    