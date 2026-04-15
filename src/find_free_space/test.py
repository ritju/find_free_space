import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer,GoalResponse,CancelResponse
from rclpy.qos import qos_profile_sensor_data, DurabilityPolicy,ReliabilityPolicy,QoSProfile,HistoryPolicy
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Point
from capella_ros_msg.srv import IsCarPassable
import tf2_ros
import numpy as np
import time
import threading
import psutil
import math
import cv2
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import Costmap
from capella_ros_msg.action import FindCarAvoidancePoint
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PolygonStamped
from collections import defaultdict
# import matplotlib.pyplot as plt
# from nav2_costmap_2d import Costmap2D

from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, QoSDurabilityPolicy


class CPUMonitor:
    def __init__(self, interval=0.02):
        self.interval = interval
        self._samples = []
        self._running = False
        self._thread = None
        self._process = psutil.Process()

    def start(self):
        self._samples = []
        self._running = True
        self._process.cpu_percent()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while self._running:
            cpu = self._process.cpu_percent()
            self._samples.append(cpu)
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
        if not self._samples:
            return 0.0, 0.0
        avg_cpu = sum(self._samples) / len(self._samples)
        peak_cpu = max(self._samples)
        return avg_cpu, peak_cpu


class CarAvoidancePointActionServer(Node):
    def __init__(self):
        super().__init__('find_free_space_action_server')
        self.get_logger().info('find_free_space_action_server started.')

        # 初始化参数
        self.robot_width = 1.0
        self.vehicle_width = 2.0
        self.search_interval = 0.5
        self.action_goal_handle_msg = None
        self.search_radius_min = 3.0
        self.search_radius_max = 4.0
        self.search_radius_extra_dis = 2.5
        self.outside_min = 0.0
        self.outside_max = 0.5
        self.cv_window_name = 'Global Costmap Raw Colored'

        self.init_params()

        self.cpu_monitor = CPUMonitor()

        # publish markers for debug
        marker_qos = QoSProfile(depth=1,
                        durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
        self.marker_car_pose_publisher = self.create_publisher(
            Marker,
            "marker_car_pose",
            marker_qos          
        )

        self.marker_robot_pose_publisher = self.create_publisher(
            Marker,
            "marker_robot_pose",
            marker_qos
        )

        self.marker_avoidance_side_publisher = self.create_publisher(
            Marker,
            "marker_avoidance_side",
            marker_qos
        )

        self.marker_searching_rect_publisher = self.create_publisher(
            Marker,
            "marker_searching_rect",
            marker_qos
        )

        self.marker_parking_point_publisher = self.create_publisher(
            Marker,
            "marker_parking_point",
            marker_qos
        )

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
        self.get_robot_pose_timer_ = None

        self.polygons = []
        self.vertices = []
        # 创建一个timer,用于实时得到距离机器人最近的通道位姿。
        # self.get_verties_timer = self.create_timer(timer_period_sec=0.5, callback=self.get_vertices_callback)        
        
        # action
        action_server_feedback_qos = QoSProfile(depth=1)
        action_server_feedback_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.action_server = ActionServer(self,FindCarAvoidancePoint,'/find_car_avoidance_point_action',
                                        self.action_goal_callback,
                                        callback_group=callback_gp1,
                                        feedback_pub_qos_profile=action_server_feedback_qos)#
        # tf2
        self.tf_buffer = None
        self.tf_listener = None
        self.global_costmap_sub = self.create_subscription(
            Costmap,
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
        self.declare_parameter("search_interval", 0.3)
        self.declare_parameter("search_radius_min", 3.0)
        self.declare_parameter("search_radius_max", 4.0)
        self.declare_parameter("search_radius_extra_dis", 2.5)
        self.declare_parameter("outside_min", 0.0)
        self.declare_parameter("outside_max", 0.5)
        self.declare_parameter("check_service_max_time", 0.5)
        self.declare_parameter('show_global_costmap_raw_cv2', False)
        self.declare_parameter('show_global_costmap_raw_colored_cv2', False)

        self.topic_name_global_costmap = self.get_parameter("topic_name_global_costmap").value
        self.service_name_check_car_passble = self.get_parameter("service_name_check_car_passble").value
        self.topic_name_footprint = self.get_parameter("topic_name_footprint").value
        self.search_interval = self.get_parameter("search_interval").value
        self.search_radius_min = self.get_parameter("search_radius_min").value
        self.search_radius_max = self.get_parameter("search_radius_max").value
        self.search_radius_extra_dis = self.get_parameter("search_radius_extra_dis").value
        self.outside_min = self.get_parameter("outside_min").value
        self.outside_max = self.get_parameter("outside_max").value
        self.check_service_max_time = self.get_parameter("check_service_max_time").value
        self.show_global_costmap_raw_cv2 = self.get_parameter('show_global_costmap_raw_cv2').value
        self.show_global_costmap_raw_colored_cv2 = self.get_parameter('show_global_costmap_raw_colored_cv2').value

        self.get_logger().info(f'topic_name_global_costmap: {self.topic_name_global_costmap}')
        self.get_logger().info(f'service_name_check_car_passble: {self.service_name_check_car_passble}')
        self.get_logger().info(f'topic_name_footprint: {self.topic_name_footprint}')
        self.get_logger().info(f'search_interval: {self.search_interval}')
        self.get_logger().info(f'search_radius_min: {self.search_radius_min}')
        self.get_logger().info(f'search_radius_max: {self.search_radius_max}')
        self.get_logger().info(f'search_radius_extra_dis: {self.search_radius_extra_dis}')
        self.get_logger().info(f'outside_min: {self.outside_min}')
        self.get_logger().info(f'outside_max: {self.outside_max}')
        self.get_logger().info(f'check_service_max_time: {self.check_service_max_time}')
        self.get_logger().info(f'show_global_costmap_raw_cv2: {self.show_global_costmap_raw_cv2}')
        self.get_logger().info(f'show_global_costmap_raw_colored_cv2: {self.show_global_costmap_raw_colored_cv2}')
    
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
        # self.get_logger().info('获取全局代价图')
        self.global_costmap = msg

        if self.show_global_costmap_raw_colored_cv2 or self.show_global_costmap_raw_cv2:
            width = msg.metadata.size_x
            height = msg.metadata.size_y
            costmap_data = np.array(msg.data, dtype=np.uint8).reshape((height, width))
        
            count0 = np.count_nonzero(costmap_data == 0)
            count254 = np.count_nonzero(costmap_data == 254)
            count255 = np.count_nonzero(costmap_data == 255)

            self.get_logger().info(f'count0: {count0}', once=True)
            self.get_logger().info(f'count254: {count254}', once=True)
            self.get_logger().info(f'count255: {count255}', once=True)
        
        if self.show_global_costmap_raw_colored_cv2:
            # 应用颜色映射（障碍物显示为红色）
            colored_map = cv2.applyColorMap(costmap_data, cv2.COLORMAP_JET)        
            # 显示图像
            cv2.imshow(self.cv_window_name, colored_map)
        
        if self.show_global_costmap_raw_cv2:
            cv2.imshow('Global Costmap Raw', costmap_data)
        
        if self.show_global_costmap_raw_colored_cv2 or self.show_global_costmap_raw_cv2:
            cv2.waitKey(1)
    
    def start_tf_listening(self):
        if self.tf_buffer is not None:
            return  # 已经在运行，不重复创建
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.get_robot_pose_timer_ = self.create_timer(0.1, self.get_robot_pose_timer_callback)
        self.get_logger().info('TF listener started')

    def stop_tf_listening(self):
        if self.get_robot_pose_timer_ is not None:
            self.get_robot_pose_timer_.cancel()
            self.destroy_timer(self.get_robot_pose_timer_)
            self.get_robot_pose_timer_ = None
        if self.tf_listener is not None:
            # 手动销毁 TransformListener 内部的订阅，防止线程泄漏
            if hasattr(self.tf_listener, 'subscription'):
                self.destroy_subscription(self.tf_listener.subscription)
            if hasattr(self.tf_listener, '_tf_static_sub'):
                self.destroy_subscription(self.tf_listener._tf_static_sub)  
            self.tf_listener = None
        if self.tf_buffer is not None:
            del self.tf_buffer
            self.tf_buffer = None
        import gc
        gc.collect()
        self.get_logger().info('TF listener stopped')

    # 用于实时获取机器人的位姿
    def get_robot_pose_timer_callback(self):
        if self.tf_buffer is None:
            return
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.robot_pose.header.stamp = self.get_clock().now().to_msg()
            self.robot_pose.header.frame_id = 'map'
            self.robot_pose.pose.position.x = trans.transform.translation.x
            self.robot_pose.pose.position.y = trans.transform.translation.y
            self.robot_pose.pose.orientation = trans.transform.rotation
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'{e}')
        
        # self.get_logger().info(f'机器人当前位姿: [{self.robot_pose.pose.position.x}, {self.robot_pose.pose.position.y}]', throttle_duration_sec=2)

    def get_vertices_callback(self):
        if len(self.polygons) == 0:
            pass
        else:
            current_polygon_vertices = []
            min_dis_robot_to_polygon = 1000.0
            for polygon in self.polygons:
                current_polygon_vertices = np.array([[point.x,point.y] for point in polygon.points])
                robot_is_in_area = self.is_point_inside_parallelogram(self.robot_pose.pose.position.x,self.robot_pose.pose.position.y,current_polygon_vertices)
                self.get_logger().info(f"robot: ({self.robot_pose.pose.position.x}, {self.robot_pose.pose.position.y})")
                self.get_logger().info(f'polygons: \n{current_polygon_vertices}')
                if robot_is_in_area:
                    self.vertices = current_polygon_vertices
                    self.get_logger().info('inside polygon: True')
                    break
                else:
                    self.get_logger().info('inside polygon: False')
                    robot_x = self.robot_pose.pose.position.x
                    robot_y = self.robot_pose.pose.position.y
                    dis = self.dis_point_to_rect(robot_x, robot_y, current_polygon_vertices)
                    if dis < min_dis_robot_to_polygon and dis < 2.0:
                        min_dis_robot_to_polygon = dis
                        self.vertices = current_polygon_vertices
                        self.get_logger().info(f'robot_to_polygon dis: {min_dis_robot_to_polygon}')
                        self.get_logger().info(f'update self.vertices: {self.vertices}')


    def dis_point_to_point(self, p1_x, p1_y, p2_x, p2_y):
        return math.sqrt(math.pow(p1_x - p2_x, 2) + math.pow(p1_y - p2_y, 2))
    
    def dis_point_to_line(self, x, y, p1_x, p1_y, p2_x, p2_y):
        """
        计算点到直线的垂直距离
        参数：
        p1_x, p1_y: 直线第一个点坐标
        p2_x, p2_y: 直线第二个点坐标
        x, y: 直线外点坐标
        返回：点到直线的距离
        """
        # 处理直线为垂直线的情况
        if p1_x == p2_x:
            return abs(x - p1_x)
        
        # 计算直线方程参数 (Ax + By + C = 0)
        A = p2_y - p1_y
        B = p1_x - p2_x
        C = p2_x * p1_y - p1_x * p2_y
        
        # 计算距离
        numerator = abs(A * x + B * y + C)
        denominator = math.sqrt(A**2 + B**2)

        dis1 = numerator / denominator
        dis2 = self.dis_point_to_point(x, y, p1_x, p1_y)
        dis3 = self.dis_point_to_point(x, y, p2_x, p2_y)
        
        return min([dis1, dis2, dis3])
    
    def dis_point_to_line2(self, x, y, p1_x, p1_y, p2_x, p2_y):
        """
        计算点到直线的垂直距离
        参数：
        p1_x, p1_y: 直线第一个点坐标
        p2_x, p2_y: 直线第二个点坐标
        x, y: 直线外点坐标
        返回：点到直线的距离
        """
        # 处理直线为垂直线的情况
        if p1_x == p2_x:
            return abs(x - p1_x)
        
        # 计算直线方程参数 (Ax + By + C = 0)
        A = p2_y - p1_y
        B = p1_x - p2_x
        C = p2_x * p1_y - p1_x * p2_y
        
        # 计算距离
        numerator = abs(A * x + B * y + C)
        denominator = math.sqrt(A**2 + B**2)

        return numerator / denominator

    def dis_point_to_rect(self, x, y, rect):
        length = len(rect)
        min_dis = 1000.0
        for i in range(length):
            p1 = rect[i]
            p2 = rect[(i+1)%length]
            dis = self.dis_point_to_line(x, y, p1[0], p1[1], p2[0], p2[1])
            if dis < min_dis:
                min_dis = dis
        return min_dis


    def action_goal_callback(self, goal_handle):
        start_perf = time.perf_counter()
        start_process = time.process_time()
        self.cpu_monitor.start()
        self.start_tf_listening()
        time.sleep(1.0)
        try:
            self.get_logger().info('开始寻找避让点...')
            # self.get_logger().info(f'goal_handle.request..{goal_handle.request}')
            self.action_goal_handle_msg = goal_handle.request
            self.vehicle_width = self.action_goal_handle_msg.car_size.y
            self.polygons = self.action_goal_handle_msg.polygons
            self.get_logger().info(f'polygons: {self.polygons}')
            
            # 获取清洁区域信息
            # 每次需要用到self.vertices时，调用一下 get_vertices_callback()
            self.get_logger().info('寻找当前通行区域...')
            # 改
            self.vertices = list(self.vertices)
            self.vertices.clear()
            self.get_vertices_callback()

            if len(self.vertices) == 0:
                self.get_logger().error('未找到用于寻找停靠点的通道')
                goal_handle.abort()
                return FindCarAvoidancePoint.Result()

            v1, v2, v3, v4 = self.vertices
            self.get_logger().info(f'当前通行区域: [({v1[0]}, {v1[1]}),({v2[0]}, {v2[1]}),({v3[0]}, {v3[1]}),({v4[0]}, {v4[1]})]')


            # self.get_logger().info(f'self.get_vertices_callback():{len(self.vertices)}')
            # self.get_logger().info(f'self.vertices:{self.vertices}')
            

            # 寻找停靠点
            self.get_logger().info('寻找停靠点...')
            avoidance_point = self.find_avoidance_point(self.robot_pose, self.vertices)
            self.get_logger().info(f'avoidance_point:{avoidance_point}')
            if avoidance_point is not None:
                self.get_logger().info(f'成功找到避让点{avoidance_point}')

                msg_marker_parking_point = Marker()
                msg_marker_parking_point.header.frame_id = "map"
                msg_marker_parking_point.header.stamp = self.get_clock().now().to_msg()
                msg_marker_parking_point.id = 5
                msg_marker_parking_point.type = Marker.ARROW
                msg_marker_parking_point.action = Marker.ADD
                msg_marker_parking_point.scale.x = 0.5
                msg_marker_parking_point.scale.y = 0.2
                msg_marker_parking_point.scale.z = 0.4
                msg_marker_parking_point.color.r = 0.0
                msg_marker_parking_point.color.g = 0.0
                msg_marker_parking_point.color.b = 1.0
                msg_marker_parking_point.color.a = 1.0
                msg_marker_parking_point.pose = avoidance_point.pose
                self.marker_parking_point_publisher.publish(msg_marker_parking_point)

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
        finally:
            wall_time = time.perf_counter() - start_perf
            cpu_time = time.process_time() - start_process
            avg_cpu, peak_cpu = self.cpu_monitor.stop()
            self.get_logger().info(
                "=== 性能统计 === 总耗时: {:.4f}s, CPU时间: {:.4f}s, CPU平均: {:.1f}%, CPU峰值: {:.1f}%".format(
                    wall_time, cpu_time, avg_cpu, peak_cpu
                )
            )
            self.stop_tf_listening()

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

    # 寻找距离机器人最近的长边    
    def find_min_long_sides(self, cleaning_area_vertices, robot_position):
        # 计算相邻顶点之间的距离
        distances = []
        for i in range(4):
            x1, y1 = cleaning_area_vertices[i]
            x2, y2 = cleaning_area_vertices[(i+1)%4]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(distance)
        
        # 判断长边对
        if distances[0] > distances[1]:
            # 长边是0-1和2-3
            long_sides = [
                (cleaning_area_vertices[0], cleaning_area_vertices[1]),
                (cleaning_area_vertices[2], cleaning_area_vertices[3])
            ]
        else:
            # 长边是1-2和3-0
            long_sides = [
                (cleaning_area_vertices[1], cleaning_area_vertices[2]),
                (cleaning_area_vertices[3], cleaning_area_vertices[0])
            ]
        
        # 计算点到线段的距离
        def point_to_line_distance(point, line):
            x0, y0 = point
            x1, y1 = line[0]
            x2, y2 = line[1]
            
            # 线段长度的平方
            l2 = (x2 - x1)**2 + (y2 - y1)**2
            
            # 如果线段实际上是一个点，返回到该点的距离
            if l2 == 0:
                return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            
            # 考虑线段参数化表示：P(t) = (1-t)*A + t*B，计算投影参数t
            t = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / l2
            
            if t < 0:
                # 投影点在A之前，返回A到点的距离
                return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            elif t > 1:
                # 投影点在B之后，返回B到点的距离
                return math.sqrt((x0 - x2)**2 + (y0 - y2)**2)
            else:
                # 投影点在线段上，计算投影点到点的距离
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                return math.sqrt((x0 - px)**2 + (y0 - py)**2)
        
        # 找出距离机器人最近的长边
        min_distance = float('inf')
        closest_side = None
        for side in long_sides:
            distance = point_to_line_distance(robot_position, side)
            if distance < min_distance:
                min_distance = distance
                closest_side = side
        
        return closest_side
    
    # 计算点与直线垂直且向外延申的点的点
    def findIntersection(self, a, b, c, distance):
        # 计算线段ab的向量
        ab_x = b[0] - a[0]
        ab_y = b[1] - a[1]
        
        # 计算线段ab的长度
        length_ab = math.sqrt(ab_x**2 + ab_y**2)
        
        # 线段ab的单位向量
        if length_ab == 0:
            # 如果线段长度为0，返回None或适当处理
            return None
        unit_ab_x = ab_x / length_ab
        unit_ab_y = ab_y / length_ab
        
        # 线段ab的法向量（垂直于线段，指向线段的右侧）
        normal_x = -unit_ab_y
        normal_y = unit_ab_x
        
        # 计算点c到线段ab所在直线的垂足
        # 向量ac
        ac_x = c[0] - a[0]
        ac_y = c[1] - a[1]
        
        # 点积计算投影长度
        projection = ac_x * unit_ab_x + ac_y * unit_ab_y
        
        # 垂足坐标
        foot_x = a[0] + projection * unit_ab_x
        foot_y = a[1] + projection * unit_ab_y
        
        # 计算向量ac在法向量上的投影（带符号）
        normal_projection = ac_x * normal_x + ac_y * normal_y
        
        # 确定方向：始终向线段的另一侧移动
        direction = -1 if normal_projection > 0 else 1
        
        # 计算目标点坐标
        target_x = foot_x + direction * normal_x * distance
        target_y = foot_y + direction * normal_y * distance
        
        return (target_x, target_y)
    
    # 寻找所有的避让点
    def find_avoidance_point(self, robot_pose, cleaning_area_vertices):
        if self.global_costmap is None:
            self.get_logger().error('全局代价地图未收到')
            return None
        # self.get_logger().error(f'全局代价图: {self.global_costmap}')
        map_info = self.global_costmap.metadata
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y
        resolution = map_info.resolution
        width = map_info.size_x
        height = map_info.size_y
        self.get_logger().info(f'origin_x: {origin_x}')
        self.get_logger().info(f'origin_y: {origin_y}')
        self.get_logger().info(f'resolution: {resolution}')
        self.get_logger().info(f'width: {width}')
        self.get_logger().info(f'height: {height}')
        
        map = np.array(self.global_costmap.data)
        # map[map == -1] = 50
        # map = (100 - map) / 100 * 255
        # map[map != 0] = 255
        costmap = np.ascontiguousarray(map.reshape((height,width)).astype(np.uint8))

        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y

        msg_marker_robot_pose = Marker()
        msg_marker_robot_pose.header.frame_id = "map"
        msg_marker_robot_pose.header.stamp = self.get_clock().now().to_msg()
        msg_marker_robot_pose.id = 2
        msg_marker_robot_pose.type = Marker.SPHERE
        msg_marker_robot_pose.action = Marker.ADD
        msg_marker_robot_pose.scale.x = 0.2
        msg_marker_robot_pose.scale.y = 0.2
        msg_marker_robot_pose.scale.z = 0.2
        msg_marker_robot_pose.color.r = 0.0
        msg_marker_robot_pose.color.g = 1.0
        msg_marker_robot_pose.color.b = 0.0
        msg_marker_robot_pose.color.a = 1.0
        msg_marker_robot_pose.pose.position.x = robot_x
        msg_marker_robot_pose.pose.position.y = robot_y
        msg_marker_robot_pose.pose.orientation.w = 1.0
        self.marker_robot_pose_publisher.publish(msg_marker_robot_pose)

        # 计算最近的边界
        
        self.get_logger().info('寻找最近的边界...')
        # 寻找距离最近的长边
        # nearest_boundary = self.find_min_long_sides(cleaning_area_vertices,[robot_x1,robot_y1])
        nearest_boundary = self.find_nearest_boundary(robot_pose, cleaning_area_vertices)
        y_ = (nearest_boundary[0][1] - nearest_boundary[1][1])
        x_ = (nearest_boundary[0][0] - nearest_boundary[1][0])
        self.get_logger().info(f'nearest_boundary: [({nearest_boundary[0][0]}, {nearest_boundary[0][1]}), ({nearest_boundary[1][0]}, {nearest_boundary[1][1]})]')
        self.get_logger().info(f'delta_y: {y_}')
        self.get_logger().info(f'delta_x: {x_}')

        msg_marker_avoidance_side = Marker()
        msg_marker_avoidance_side.header.frame_id = "map"
        msg_marker_avoidance_side.header.stamp = self.get_clock().now().to_msg()
        msg_marker_avoidance_side.id = 3
        msg_marker_avoidance_side.type = Marker.LINE_LIST
        msg_marker_avoidance_side.action = Marker.ADD
        msg_marker_avoidance_side.scale.x = 0.2
        msg_marker_avoidance_side.color.r = 0.0
        msg_marker_avoidance_side.color.g = 1.0
        msg_marker_avoidance_side.color.b = 0.0
        msg_marker_avoidance_side.color.a = 1.0
        point1 = Point()
        point1.x = nearest_boundary[0][0]
        point1.y = nearest_boundary[0][1]
        msg_marker_avoidance_side.points.append(point1)
        point2 = Point()
        point2.x = nearest_boundary[1][0]
        point2.y = nearest_boundary[1][1]
        msg_marker_avoidance_side.points.append(point2)
        
        self.marker_avoidance_side_publisher.publish(msg_marker_avoidance_side)

        # if abs(x_) < 1e-3:
        #     self.get_logger().info(f'delta_x == 0: true')
        #     k = math.pi /2
        # else:
        #     self.get_logger().info(f'delta_x == 0: false')
        #     k = y_ / x_
        #     self.get_logger().info(f'k: {k}')
        #     k = np.arctan2(k)
        k = np.arctan2(y_, x_)
        p1,p2 = nearest_boundary        
        self.get_logger().info(f'k_radian: {k}')
        self.get_logger().info(f'k_degree: {k/math.pi*180}')
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
        self.get_logger().info(f'robot: ({robot_x}, {robot_y})')
        self.get_logger().info(f'car: ({car_pose.x}, {car_pose.y})')

        msg_marker_car_pose = Marker()
        msg_marker_car_pose.header.frame_id = "map"
        msg_marker_car_pose.header.stamp = self.get_clock().now().to_msg()
        msg_marker_car_pose.id = 1
        msg_marker_car_pose.type = Marker.CUBE
        msg_marker_car_pose.action = Marker.ADD
        msg_marker_car_pose.scale.x = 0.5
        msg_marker_car_pose.scale.y = 0.5
        msg_marker_car_pose.scale.z = 0.5
        msg_marker_car_pose.color.r = 1.0
        msg_marker_car_pose.color.g = 0.0
        msg_marker_car_pose.color.b = 0.0
        msg_marker_car_pose.color.a = 1.0
        msg_marker_car_pose.pose.position.x = car_pose.x
        msg_marker_car_pose.pose.position.y = car_pose.y
        msg_marker_car_pose.pose.orientation.w = 1.0
        self.marker_car_pose_publisher.publish(msg_marker_car_pose)

        y_ = robot_y - car_pose.y
        x_ = robot_x - car_pose.x
        # if x_ == 0.0:
        #     car_robot_k = math.pi /2
        # else:
        #     car_robot_k = y_ / x_
        #     car_robot_k = np.arctan2(car_robot_k)
        car_robot_k = np.arctan2(y_, x_)
        self.get_logger().info(f"k: {k}, car_robot_k: {car_robot_k}")
        k_diff = self.angle_diff(k,car_robot_k)
        self.get_logger().info(f'k_diff: {k_diff}')
        if k_diff > math.pi/2:
            k = self.add_angles(k,math.pi)

        # 计算避让方向k和机器人方向的夹角大小 => 夹角小于pi/2.0,认为二者方向相同;大于pi/2.0,认为二者方向不同。
        # k_robot = np.arctan2(robot_y, robot_x)  # error
        orientation = robot_pose.pose.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
    
        # 将四元数转换为欧拉角(roll, pitch, yaw)
        (roll, pitch, yaw) = euler_from_quaternion(quaternion)
        k_robot = yaw
        k_diff2 = self.angle_diff(k, k_robot)
        self.get_logger().info(f'k: {k}, k_robot: {k_robot}, k_diff2: {k_diff2}')
        robot_x1 = 0.0
        robot_y1 = 0.0
        robot_x2 = 0.0
        robot_y2 = 0.0
        if k_diff2 < math.pi / 2:
            self.get_logger().info(f'在机器人前方避车, 额外增加{self.search_radius_extra_dis}米搜索距离')
            robot_x1 = robot_x + math.cos(k) * (self.search_radius_max + self.search_radius_extra_dis)
            robot_y1 = robot_y + math.sin(k) * (self.search_radius_max + self.search_radius_extra_dis)
            robot_x2 = robot_x + math.cos(k) * (self.search_radius_min + self.search_radius_extra_dis)
            robot_y2 = robot_y + math.sin(k) * (self.search_radius_min + self.search_radius_extra_dis)
        else:
            self.get_logger().info(f'在机器人后方避车')
            robot_x1 = robot_x + math.cos(k) * self.search_radius_max
            robot_y1 = robot_y + math.sin(k) * self.search_radius_max
            robot_x2 = robot_x + math.cos(k) * self.search_radius_min
            robot_y2 = robot_y + math.sin(k) * self.search_radius_min

        # robot_x1 = robot_x + math.cos(k) * self.search_radius_max
        # robot_y1 = robot_y + math.sin(k) * self.search_radius_max
        # robot_x2 = robot_x + math.cos(k) * self.search_radius_min
        # robot_y2 = robot_y + math.sin(k) * self.search_radius_min    

        
        if (self.is_point_inside_parallelogram(robot_x, robot_y, self.vertices)):
            vertical_border_x1, vertical_border_y1 = self.findIntersection(nearest_boundary[0],nearest_boundary[1],[robot_x1,robot_y1], self.outside_max)
            vertical_border_x2, vertical_border_y2 = self.findIntersection(nearest_boundary[0],nearest_boundary[1],[robot_x2,robot_y2], self.outside_max)
            
            robot_x1, robot_y1 = self.findIntersection(nearest_boundary[0],nearest_boundary[1],[robot_x1,robot_y1], self.outside_min)
            robot_x2, robot_y2 = self.findIntersection(nearest_boundary[0],nearest_boundary[1],[robot_x2,robot_y2], self.outside_min)
        else:
            dis_robot_to_nearest_bound = self.dis_point_to_line2(robot_x, robot_y, nearest_boundary[0][0], nearest_boundary[0][1], nearest_boundary[1][0], nearest_boundary[1][1])
            if dis_robot_to_nearest_bound > self.outside_min and dis_robot_to_nearest_bound < self.outside_max:
                self.outside_min = dis_robot_to_nearest_bound
            elif dis_robot_to_nearest_bound >= self.outside_max:
                self.get_logger().info("返回机器人当前点为停靠点")
                ret_pose = PoseStamped()
                ret_pose.header.stamp = self.get_clock().now().to_msg()
                ret_pose.header.frame_id = "map"
                ret_pose.pose.position.x = robot_x
                ret_pose.pose.position.y = robot_y
                target_angle = math.degrees(math.atan2(nearest_boundary[1][1] - nearest_boundary[0][1], nearest_boundary[1][0] - nearest_boundary[0][0]))
                yaw = self.get_yaw_from_pose(robot_pose)
                ret_pose_yaw = math.radians(self.adjust_angle((math.cos(yaw), math.sin(yaw)), target_angle))
                quat = Quaternion()
                quat.x, quat.y, quat.z, quat.w = quaternion_from_euler(0, 0, ret_pose_yaw)
                ret_pose.pose.orientation = quat
                return ret_pose


            vertical_border_x1, vertical_border_y1 = self.findIntersection(nearest_boundary[0],nearest_boundary[1],[robot_x1,robot_y1], -self.outside_max)
            vertical_border_x2, vertical_border_y2 = self.findIntersection(nearest_boundary[0],nearest_boundary[1],[robot_x2,robot_y2], -self.outside_max)
            
            robot_x1, robot_y1 = self.findIntersection(nearest_boundary[0],nearest_boundary[1],[robot_x1,robot_y1], -self.outside_min)
            robot_x2, robot_y2 = self.findIntersection(nearest_boundary[0],nearest_boundary[1],[robot_x2,robot_y2], -self.outside_min)
        
        # 在区域内搜索，往边界靠近
        # 四个点按照顺序排序
        find_vertices = self.sort_quadrilateral_vertices([(robot_x1,robot_y1),(vertical_border_x1,vertical_border_y1),(vertical_border_x2, vertical_border_y2),(robot_x2,robot_y2)])
        
        self.get_logger().info('生成所有停靠点...')
        self.get_logger().info(f"find_vertices: {find_vertices}")
        
        msg_marker_searching_rect = Marker()
        msg_marker_searching_rect.header.frame_id = "map"
        msg_marker_searching_rect.header.stamp = self.get_clock().now().to_msg()
        msg_marker_searching_rect.id = 4
        msg_marker_searching_rect.type = Marker.LINE_LIST
        msg_marker_searching_rect.action = Marker.ADD
        msg_marker_searching_rect.scale.x = 0.1
        msg_marker_searching_rect.color.r = 1.0
        msg_marker_searching_rect.color.g = 0.0
        msg_marker_searching_rect.color.b = 0.0
        msg_marker_searching_rect.color.a = 1.0
        size_tmp = len(find_vertices)
        for i in range(size_tmp):
            p_start = Point()
            p_start.x = find_vertices[i][0]
            p_start.y = find_vertices[i][1]
            msg_marker_searching_rect.points.append(p_start)
            p_end = Point()
            p_end.x = find_vertices[(i+1)%size_tmp][0]
            p_end.y = find_vertices[(i+1)%size_tmp][1]
            msg_marker_searching_rect.points.append(p_end)
        self.marker_searching_rect_publisher.publish(msg_marker_searching_rect)

        search_posestamped_list = self.select_points_in_parallelogram(find_vertices,0.05, k)
        self.get_logger().info(f'search_posestamped length: {len(search_posestamped_list)}')
        # 判断每个点是否里障碍物太近
        # 首先将位姿转换到map的像素点
        if len(search_posestamped_list) > 0:
            boundary_points = [(x, y) for (x, y, _) in search_posestamped_list]
            boundary_points = np.array(boundary_points)
            boundary_points_pixel = (boundary_points - np.array([origin_x,origin_y])) / resolution
            # boundary_points_pixel[:,1] = height - boundary_points_pixel[:,1]
            boundary_points_pixel[:,0] = np.clip(boundary_points_pixel[:,0],0,width-1)
            boundary_points_pixel[:,1] = np.clip(boundary_points_pixel[:,1],0,height-1)
            self.get_logger().info(f'一共{len(boundary_points)}个避障...')
            # # 判断目标点附近是否有障碍物
            self.get_logger().info('排除障碍物点...')
            is_obstacle_index = [True if self.check_point_is_free(costmap,(x,y),2) else False for x,y in boundary_points_pixel]
            boundary_points = boundary_points[is_obstacle_index]
            search_posestamped_list = [search_posestamped_list[i] for i in range(len(search_posestamped_list)) if is_obstacle_index[i]]
            self.get_logger().info(f'排除障碍物点后还剩{len(search_posestamped_list)}个避障...')
            
            robot_point = np.array([robot_x,robot_y])
            robot_point_pixel = (robot_point - np.array([origin_x,origin_y])) / resolution
            # robot_point_pixel[1] = height - robot_point_pixel[1]
            robot_point_pixel[0] = np.clip(robot_point_pixel[0],0,width-1)
            robot_point_pixel[1] = np.clip(robot_point_pixel[1],0,height-1)
            # robot_x_p,robot_y_p = robot_point_pixel
            robot_x_p = int(robot_point_pixel[0])
            robot_y_p = int(robot_point_pixel[1])

            for avoidance_pose in search_posestamped_list:    
                # avoidance_pose_msg = IsCarPassable.Request()
                # avoidance_pose_msg.robot_pose = avoidance_pose
                # avoidance_pose_msg.car_pose = self.action_goal_handle_msg.car_pose
                # avoidance_pose_msg.size = self.action_goal_handle_msg.car_size
                # self.get_logger().info(f'避让点: ({avoidance_pose[0]}, {avoidance_pose[1]})')
                # start_time = time.time()
                # check_avoidance_result = self.check_avoidance(avoidance_pose_msg)
                # end_time = time.time()
                # delta_time = end_time - start_time
                # self.get_logger().info(f'check_avoidance_result: {check_avoidance_result}')
                # self.get_logger().info(f'delta_time: {delta_time}')
                # if check_avoidance_result and delta_time < self.check_service_max_time:

                point = np.array([avoidance_pose[0], avoidance_pose[1]])
                point_pixel = (point - np.array([origin_x,origin_y])) / resolution
                # point_pixel[1] = height - point_pixel[1]
                point_pixel[0] = np.clip(point_pixel[0],0,width-1)
                point_pixel[1] = np.clip(point_pixel[1],0,height-1)
                # point_x_p,point_y_p = point_pixel
                point_x_p = int(point_pixel[0])
                point_y_p = int(point_pixel[1])
                
                # ========== 改动3b：修改 bresenham 调用方式 ==========
                bresenham_result = self.bresenham_check(robot_x_p, robot_y_p, point_x_p, point_y_p, costmap)
                if bresenham_result:
                    self.get_logger().info('机器人到当前点的连线满足')
                    ret_pose = PoseStamped()
                    ret_pose.header.stamp = self.get_clock().now().to_msg()
                    ret_pose.header.frame_id = 'map'
                    ret_pose.pose.position.x = avoidance_pose[0]
                    ret_pose.pose.position.y = avoidance_pose[1]
                    quat = Quaternion()
                    quat.x, quat.y, quat.z, quat.w = quaternion_from_euler(0, 0, math.radians(avoidance_pose[2]))
                    ret_pose.pose.orientation = quat
                    return ret_pose
                else:
                    self.get_logger().info('机器人到当前点的连线不满足')
                # ========== 改动3b结束 ==========
        else:
            self.get_logger().info('用于搜索的点，数量为0')
            return None
        self.get_logger().info('所有点都不满足')
        return None

    def get_yaw_from_pose(self, pose_stamped):
        """从PoseStamped消息中提取yaw角"""
        orientation = pose_stamped.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(quaternion)
        return yaw

    # 寻找最近的边界
    def find_nearest_boundary(self, robot_pose, vertices):
        """
        找到距离机器人位置最近的矩形长边
        
        参数:
            robot_pose: 包含机器人位置信息的对象
            vertices: 矩形四个顶点坐标列表，按顺序排列
            
        返回:
            距离机器人最近的边界线段（由两个端点坐标组成的元组）
        """
        # 构建四条边（假设顶点已按顺序排列）
        edges = [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]), 
            (vertices[2], vertices[3]),
            (vertices[3], vertices[0])
        ]
        
        # 计算各边长度
        def calc_edge_length(edge):
            p1, p2 = edge
            return np.linalg.norm(np.array(p2) - np.array(p1))
        
        edge_lengths = [calc_edge_length(edge) for edge in edges]
        
        # 按长度排序并获取两条最长边
        sorted_edges = sorted(zip(edges, edge_lengths), key=lambda x: -x[1])
        long_edges = [edge for edge, _ in sorted_edges[:2]]
        
        # 机器人当前位置
        robot_point = np.array([robot_pose.pose.position.x, robot_pose.pose.position.y])
        
        min_distance = float('inf')
        nearest_boundary = None
        
        # 计算到每条长边的距离
        for edge in long_edges:
            p1, p2 = np.array(edge[0]), np.array(edge[1])
            
            # 计算直线方程参数 ax + by + c = 0
            a = p1[1] - p2[1]
            b = p2[0] - p1[0] 
            c = p1[0]*p2[1] - p2[0]*p1[1]
            
            # 计算点到直线距离
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
        使用射线法判断点是否在多边形内部
        :param robot_x: 点的x坐标
        :param robot_y: 点的y坐标
        :param vertices: 多边形顶点列表，格式为[(x1,y1), (x2,y2), ...]
        :return: True(在内部)或False(在外部)
        """
        inside = False
        j = len(vertices) - 1
        
        for i in range(len(vertices)):
            xi, yi = vertices[i][0], vertices[i][1]
            xj, yj = vertices[j][0], vertices[j][1]
            
            # 判断边是否与从(robot_x, robot_y)出发的水平射线相交
            intersect = ((yi > robot_y) != (yj > robot_y)) and \
                        (robot_x < (xj - xi) * (robot_y - yi) / (yj - yi) + xi)
                        # ((xj - xi) * (robot_y - yi) - (yj - yi) * (robot_x - xi)) > 0  # error
            
            self.get_logger().info(f'i: {i}, j: {j}')
            self.get_logger().info(f'robot_x: {robot_x}, robot_y: {robot_y}')
            self.get_logger().info(f'xi: {xi}, yi: {yi}')
            self.get_logger().info(f'xj: {xj}, yi: {yj}')
            if intersect:
                self.get_logger().info(f'intersect: True')
                inside = not inside
            else:
                self.get_logger().info(f'intersect: False')
            j = i  # 更新j为当前i，用于下一次迭代
        
        return inside
    
    # 选择平行四边形区域内的点
    def select_points_in_parallelogram(self, vertices, interval, direction):
        generate_search_points_without_directions = self.generate_all_search_points(vertices, interval)
        generate_search_points_with_directions = self.process_points(self.robot_pose, vertices, generate_search_points_without_directions, direction)
        search_posetampd_list = []
        for point_with_direction in generate_search_points_with_directions:
            x = point_with_direction[0][0]
            y = point_with_direction[0][1]
            angle = point_with_direction[1]
            search_posetampd_list.append((x, y, angle))
        # self.show(generate_search_points_with_directions)
        return search_posetampd_list

    # ========== 改动1：优化 generate_all_search_points 方法 ==========
    def generate_all_search_points(self, vertices, interval):
        A = np.array(vertices[0])
        B = np.array(vertices[1])
        D = np.array(vertices[3])
        
        u = B - A  
        v = D - A  
        
        u_length = np.linalg.norm(u)
        v_length = np.linalg.norm(v)
        steps_u = max(1, int(u_length / interval))
        steps_v = max(1, int(v_length / interval))
        
        # 向量化生成网格点
        i_vals = np.arange(steps_u + 1)
        j_vals = np.arange(steps_v + 1)
        ii, jj = np.meshgrid(i_vals, j_vals, indexing='ij')
        
        # 向量化计算所有点坐标
        px = (A[0] + u[0] * ii / steps_u + v[0] * jj / steps_v).ravel()
        py = (A[1] + u[1] * ii / steps_u + v[1] * jj / steps_v).ravel()
        
        # 向量化计算距离并排序
        robot_x = self.robot_pose.pose.position.x
        robot_y = self.robot_pose.pose.position.y
        dist_sq = (px - robot_x)**2 + (py - robot_y)**2
        idx = np.argsort(dist_sq)
        
        # 返回排序后的点列表
        return [(float(px[i]), float(py[i])) for i in idx]
    # ========== 改动1结束 ==========

    def distance_sq(self, point):
        dx = point[0] - self.robot_pose.pose.position.x
        dy = point[1] - self.robot_pose.pose.position.y
        return dx**2 + dy**2

    def calculate_long_edges(self, vertices):
        edges = []
        for i in range(4):
            p1 = vertices[i]
            p2 = vertices[(i+1)%4]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            edges.append((length, angle, (p1,p2)))
        sorted_edges = sorted(edges, key=lambda x: -x[0])
        return sorted_edges[0], sorted_edges[1]

    def distance_point_to_line(self, point, line):
        (x0, y0), (x1, y1) = line
        px, py = point
        line_length = math.hypot(x1-x0, y1-y0)
        if line_length == 0:
            return math.hypot(px-x0, py-y0)
        
        t = ((px-x0)*(x1-x0) + (py-y0)*(y1-y0)) / line_length**2
        t = max(0, min(1, t))
        proj_x = x0 + t*(x1-x0)
        proj_y = y0 + t*(y1-y0)
        return math.hypot(px-proj_x, py-proj_y)

    def adjust_angle(self, direction_vec, target_angle):

        if (target_angle > 0):
                target_angle_2 = target_angle - 180
        else:
                target_angle_2 = 180 + target_angle

        v_target_angle =   (math.cos(math.radians(target_angle)),   math.sin(math.radians(target_angle)))
        v_target_angle_2 = (math.cos(math.radians(target_angle_2)), math.sin(math.radians(target_angle_2)))

        # v_target_angle =   (math.cos(target_angle),   math.sin(target_angle))
        # v_target_angle_2 = (math.cos(target_angle_2), math.sin(target_angle_2))

        direction_vec = np.array(direction_vec)
        v_target_angle = np.array(v_target_angle)
        v_target_angle_2 = np.array(v_target_angle_2)


        if np.linalg.norm(direction_vec) == 0 or np.linalg.norm(v_target_angle) == 0 or np.linalg.norm(v_target_angle_2) == 0:
            return target_angle

        cos_theta_1 = np.dot(direction_vec, v_target_angle) / (np.linalg.norm(direction_vec)*np.linalg.norm(v_target_angle))
        cos_theta_1 = np.clip(cos_theta_1, -1.0, 1.0)  

        cos_theta_2 = np.dot(direction_vec, v_target_angle_2) / (np.linalg.norm(direction_vec)*np.linalg.norm(v_target_angle_2))
        cos_theta_2 = np.clip(cos_theta_2, -1.0, 1.0)  

        angle_ret = target_angle if np.abs(np.arccos(cos_theta_1)) < np.abs(np.arccos(cos_theta_2)) else target_angle_2
        # self.get_logger().info(f'target_angle: {target_angle}')
        # self.get_logger().info(f'target_angle2: {target_angle_2}')
        # self.get_logger().info(f'np.arccos(cos_theta_1): {np.arccos(cos_theta_1)}')
        # self.get_logger().info(f'np.arccos(cos_theta_2: {np.arccos(cos_theta_2)}')
        # self.get_logger().info(f'angle_ret: {angle_ret}')

        return target_angle if np.abs(np.arccos(cos_theta_1)) < np.abs(np.arccos(cos_theta_2)) else target_angle_2
    
    def process_points(self, robot_pose, vertices, points, direction):
        results = []
        for point in points:
            dx = point[0] - robot_pose.pose.position.x
            dy = point[1] - robot_pose.pose.position.y
            alpha = math.degrees(math.atan2(dy, dx))
            
            target_angle = math.degrees(direction)
            direction_vec = (dx, dy)
            alpha = self.adjust_angle(direction_vec, target_angle)
            results.append((point, alpha))
        
        return results
    
    # def show(self, points, arrow_length=0.5):
    #     plt.figure(figsize=(10, 8))
    #     ax = plt.gca()
        
    #     # 提取所有坐标点
    #     coordinates = np.array([p[0] for p in points])
    #     x_min, x_max = coordinates[:,0].min(), coordinates[:,0].max()
    #     y_min, y_max = coordinates[:,1].min(), coordinates[:,1].max()
        
    #     # 计算动态箭头长度（基于坐标范围）
    #     axis_range = max(x_max-x_min, y_max-y_min) * 0.2
    #     scale_factor = arrow_length * axis_range

    #     # 绘制每个点及角度箭头
    #     for (x, y), alpha in points:
    #         # 绘制坐标点[5](@ref)
    #         plt.scatter(x, y, c='red', s=80, edgecolor='black', zorder=3)
            
    #         # 计算箭头方向向量[2,8](@ref)
    #         dx = scale_factor * np.cos(np.deg2rad(alpha))
    #         dy = scale_factor * np.sin(np.deg2rad(alpha))
            
    #         # 绘制角度箭头[1,6](@ref)
    #         ax.annotate(
    #             '', 
    #             xytext=(x, y),  # 起点
    #             xy=(x+dx, y+dy),  # 终点
    #             arrowprops=dict(
    #                 arrowstyle='->',
    #                 linewidth=2,
    #                 color='blue',
    #                 mutation_scale=20,
    #                 shrinkA=0,  # 取消起点收缩
    #                 shrinkB=0   # 取消终点收缩
    #             ),
    #             zorder=2
    #         )
            
    #         # 添加角度文本标注[3,8](@ref)
    #         text_x = x + dx * 1.2
    #         text_y = y + dy * 1.2
    #         plt.text(text_x, text_y, 
    #                 f'{alpha}°', 
    #                 fontsize=10, 
    #                 color='darkgreen',
    #                 ha='center', 
    #                 va='center')

    #     # 设置坐标轴
    #     plt.grid(linestyle='--', alpha=0.7)
    #     plt.xlabel('X Axis')
    #     plt.ylabel('Y Axis')
    #     plt.title('Points with Directional Arrows')
    #     plt.axis('equal')  # 等比例坐标轴
    #     plt.show()

    # 对四边形的四个顶点进行排序，按左上角、右上角、右下角、左下角的顺序返回
    def sort_quadrilateral_vertices(self, points):
        """
        对矩形的四个顶点进行排序，返回顺序为 [左上, 右上, 右下, 左下]
        
        参数:
            points (np.ndarray or list): 四个点的坐标，形状为 (4, 2)
        
        返回:
            np.ndarray: 排序后的四个点，形状为 (4, 2)
        """
        points = np.array(points)
        
        # 1. 计算中心点
        center = np.mean(points, axis=0)
        
        # 2. 计算每个点相对于中心的角度（使用反正切函数）
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        
        # 3. 按角度排序（顺时针方向）
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        # 4. 确保顺序是 [左上, 右上, 右下, 左下]
        # 找到 y 值最小的两个点（顶部点）
        top_points = sorted_points[np.argsort(sorted_points[:, 1])[:2]]
        
        # 在顶部点中，x 较小的为左上，较大的为右上
        if top_points[0][0] > top_points[1][0]:
            top_points = top_points[::-1]
        
        # 剩下的两个点是底部点，x 较大的为右下，较小的为左下
        bottom_points = sorted_points[np.argsort(sorted_points[:, 1])[2:]]
        if bottom_points[0][0] < bottom_points[1][0]:
            bottom_points = bottom_points[::-1]
        
        # 组合最终顺序
        ordered_points = np.vstack([top_points, bottom_points])
        
        return ordered_points
    
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
    
    # ========== 改动2：优化 check_point_is_free 方法 ==========
    def check_point_is_free(self, image, center, radius=20):
        """
        判断以指定像素点为中心、半径为radius 像素的圆形区域内所有像素值是否都等于0
        :param image: 输入的单通道图像（灰度图）
        :param center: 中心像素点的坐标 (x, y)
        :param radius: 圆形区域的半径
        :return: 如果圆形区域内所有像素值都等于 0 返回 True，否则返回 False
        """
        height, width = image.shape
        r_sq = radius * radius
        cx = int(center[0])
        cy = int(center[1])
        
        y_start = max(0, cy - radius)
        y_end = min(height - 1, cy + radius)
        x_start = max(0, cx - radius)
        x_end = min(width - 1, cx + radius)
        
        for y in range(y_start, y_end + 1):
            dy = y - cy
            dy_sq = dy * dy
            for x in range(x_start, x_end + 1):
                dx = x - cx
                if dx * dx + dy_sq <= r_sq:
                    if image[y, x] == 254:
                        return False
        return True
    # ========== 改动2结束 ==========

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
        # self.get_logger().info(f'm_x: {len(map_array[0])}, y: {len(map_array)}')
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

    # ========== 改动3a：新增 bresenham_check 方法 ==========
    def bresenham_check(self, current_x, current_y, target_x, target_y, map_array):
        """
        检查两点连线是否通畅（无障碍物）
        :param current_x: 当前点的 x 坐标
        :param current_y: 当前点的 y 坐标
        :param target_x: 目标点的 x 坐标
        :param target_y: 目标点的 y 坐标
        :param map_array: 地图数组
        :return: 如果连线无障碍物返回 True，否则返回 False
        """
        dx = abs(target_x - current_x)
        dy = abs(target_y - current_y)
        sx = 1 if current_x < target_x else -1
        sy = 1 if current_y < target_y else -1
        err = dx - dy

        x, y = current_x, current_y
        
        while True:
            # 检查点是否在地图范围内且无障碍物
            if 0 <= x < len(map_array[0]) and 0 <= y < len(map_array):
                if map_array[y, x] > 253:
                    return False
            
            if x == target_x and y == target_y:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err = err - dy
                x = x + sx
            if e2 < dx:
                err = err + dx
                y = y + sy

        return True
    # ========== 改动3a结束 ==========

def main(args=None):
    rclpy.init(args=args)
    executor_ = MultiThreadedExecutor()
    car_avoidance_action_server = CarAvoidancePointActionServer()
    rclpy.spin(car_avoidance_action_server,executor=executor_)
    car_avoidance_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()