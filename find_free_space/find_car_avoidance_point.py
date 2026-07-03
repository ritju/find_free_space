import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer,GoalResponse,CancelResponse
from rclpy.qos import qos_profile_sensor_data, DurabilityPolicy,ReliabilityPolicy,QoSProfile,HistoryPolicy
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Point, PoseArray
from capella_ros_msg.srv import IsCarPassable
from garage_utils_msgs.msg import Polygons
import tf2_ros
import numpy as np
import time
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
        self.footprint_vertices = []
        self.cv_window_name = 'Global Costmap Raw Colored'
        self._static_check_info_logged = False

        self.init_params()

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

        self.marker_special_terrain_publisher = self.create_publisher(
            Marker,
            "marker_special_terrain",
            marker_qos
        )

        self.marker_all_passages_publisher = self.create_publisher(
            Marker,
            "marker_all_passages",
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
        self.get_robot_pose_timer_ = self.create_timer(timer_period_sec=0.1, callback=self.get_robot_pose_timer_callback)

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
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.global_costmap_sub = self.create_subscription(
            Costmap,
            self.topic_name_global_costmap,
            self.global_costmap_callback,
            10,
            callback_group=callback_gp2)
        self.global_costmap = None
        # 检查pose能否避让的服务
        self.check_avoidance_service = self.create_client(IsCarPassable, '/check_car_passable',callback_group=callback_gp3)

        # 禁扫区多边形（/cleaning_tool_retraction_areas）
        self.special_terrain_polygons = None
        qos_transient = QoSProfile(depth=1)
        qos_transient.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.special_terrain_sub = self.create_subscription(
            Polygons,
            '/cleaning_tool_retraction_areas',
            self._special_terrain_callback,
            qos_transient,
        )

        # 避车停车点
        self.vehicle_avoidance_stop_points = None
        qos_stop_points = QoSProfile(depth=1)
        qos_stop_points.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_stop_points.reliability = ReliabilityPolicy.RELIABLE
        self.vehicle_stop_points_sub = self.create_subscription(
            PoseArray,
            '/vehicle_avoidance_stop_points',
            self._vehicle_stop_points_callback,
            qos_stop_points,
        )

    def init_params(self):
        self.declare_parameter("topic_name_global_costmap", "")          # 全局代价地图话题名
        self.declare_parameter("service_name_check_car_passble", "")     # 车辆可通过性检查服务名
        self.declare_parameter("topic_name_footprint", "")               # 机器人足迹话题名
        self.declare_parameter("search_interval", 0.3)                   # 搜索角度间隔
        self.declare_parameter("search_radius_min", 3.0)                 # 搜索半径最小值
        self.declare_parameter("search_radius_max", 4.0)                 # 搜索半径最大值
        self.declare_parameter("search_radius_extra_dis", 2.5)           # 搜索半径额外增加距离
        self.declare_parameter("outside_min", 0.0)                       # 搜索框外侧最小偏移
        self.declare_parameter("outside_max", 0.5)                       # 搜索框外侧最大偏移
        self.declare_parameter("inside_step", 0.1)                       # 每轮往内移动步长
        self.declare_parameter('inward_offset', 0.0)                     # 搜索框初始位置离多边形边线往内多少米（0=紧贴边线）
        self.declare_parameter('max_inward_offset', 1.5)                 # 搜索框最大往内偏移距离
        self.declare_parameter("check_service_max_time", 0.5)            # 车辆通过性检查超时
        self.declare_parameter('show_global_costmap_raw_cv2', False)     # 显示原始代价地图
        self.declare_parameter('show_global_costmap_raw_colored_cv2', False)  # 显示彩色代价地图

        self.declare_parameter("search_point_interval", 0.15)            # 下采样的阈值，越大点越少
        self.declare_parameter('footprint_sweep_long_step', 0.0)       # 扫掠检查长边步长(m)，默认999.0(>footprint长边)=不启用;设<=长边才启用;设0=全像素填充
        self.declare_parameter('footprint_sweep_short_step', 0.0)      # 扫掠检查短边步长(m)，默认999.0(>footprint短边)=只扫两条长边;设<=短边=网格采样;设0=全像素填充
        self.declare_parameter('external_point_max_distance', 8.0)      # 外部点最大距离
        self.declare_parameter('point_free_check_radius', 0.2)          # 候选点周围无障碍检查半径(m)
        self.declare_parameter('stop_in_place_min_dist', 2.0)           # 机器人在通道外长边方向、且离长边超过这个距离才允许原地停

        self.topic_name_global_costmap = self.get_parameter("topic_name_global_costmap").value
        self.service_name_check_car_passble = self.get_parameter("service_name_check_car_passble").value
        self.topic_name_footprint = self.get_parameter("topic_name_footprint").value
        self.search_interval = self.get_parameter("search_interval").value
        self.search_radius_min = self.get_parameter("search_radius_min").value
        self.search_radius_max = self.get_parameter("search_radius_max").value
        self.search_radius_extra_dis = self.get_parameter("search_radius_extra_dis").value
        self.outside_min = self.get_parameter("outside_min").value
        self.outside_max = self.get_parameter("outside_max").value
        self.inside_step = self.get_parameter("inside_step").value
        self.check_service_max_time = self.get_parameter("check_service_max_time").value
        self.show_global_costmap_raw_cv2 = self.get_parameter('show_global_costmap_raw_cv2').value
        self.show_global_costmap_raw_colored_cv2 = self.get_parameter('show_global_costmap_raw_colored_cv2').value
        self.inward_offset = self.get_parameter('inward_offset').value
        self.max_inward_offset = self.get_parameter('max_inward_offset').value
        self.search_point_interval = self.get_parameter("search_point_interval").value
        self.footprint_sweep_long_step = self.get_parameter('footprint_sweep_long_step').value
        self.footprint_sweep_short_step = self.get_parameter('footprint_sweep_short_step').value
        self.external_point_max_distance = self.get_parameter('external_point_max_distance').value
        self.point_free_check_radius = self.get_parameter('point_free_check_radius').value
        self.stop_in_place_min_dist = self.get_parameter('stop_in_place_min_dist').value

        self.get_logger().info(f'topic_name_global_costmap: {self.topic_name_global_costmap}')
        self.get_logger().info(f'service_name_check_car_passble: {self.service_name_check_car_passble}')
        self.get_logger().info(f'topic_name_footprint: {self.topic_name_footprint}')
        self.get_logger().info(f'search_interval: {self.search_interval}')
        self.get_logger().info(f'search_radius_min: {self.search_radius_min}')
        self.get_logger().info(f'search_radius_max: {self.search_radius_max}')
        self.get_logger().info(f'search_radius_extra_dis: {self.search_radius_extra_dis}')
        self.get_logger().info(f'outside_min: {self.outside_min}')
        self.get_logger().info(f'outside_max: {self.outside_max}')
        self.get_logger().info(f'inside_step: {self.inside_step}')
        self.get_logger().info(f'check_service_max_time: {self.check_service_max_time}')
        self.get_logger().info(f'show_global_costmap_raw_cv2: {self.show_global_costmap_raw_cv2}')
        self.get_logger().info(f'show_global_costmap_raw_colored_cv2: {self.show_global_costmap_raw_colored_cv2}')
        self.get_logger().info(f'inward_offset: {self.inward_offset}')
        self.get_logger().info(f'max_inward_offset: {self.max_inward_offset}')
        self.get_logger().info(f'search_point_interval: {self.search_point_interval}')
        self.get_logger().info(f'footprint_sweep_long_step: {self.footprint_sweep_long_step}  # 默认999.0=不启用扫掠检查')
        self.get_logger().info(f'footprint_sweep_short_step: {self.footprint_sweep_short_step} # 默认999.0=只扫两条长边')
        self.get_logger().info(f'external_point_max_distance: {self.external_point_max_distance}')
        self.get_logger().info(f'point_free_check_radius: {self.point_free_check_radius}')
        self.get_logger().info(f'stop_in_place_min_dist: {self.stop_in_place_min_dist}')
    
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
        # 用边长构造以机器人中心为原点的局部坐标 footprint
        half_long = np.max(edges_distance) / 2.0
        half_short = np.min(edges_distance) / 2.0
        self.footprint_vertices = [
            (half_long, half_short),
            (half_long, -half_short),
            (-half_long, -half_short),
            (-half_long, half_short)
        ]
        fp_str = ', '.join([f'({v[0]:.3f}, {v[1]:.3f})' for v in self.footprint_vertices])
        self.get_logger().info(f"robot_width: {self.robot_width:.3f}, footprint_vertices (local): [{fp_str}]")
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
            self.get_logger().error(f'{e}', throttle_duration_sec=5.0)
        
        # self.get_logger().info(f'机器人当前位姿: [{self.robot_pose.pose.position.x}, {self.robot_pose.pose.position.y}]', throttle_duration_sec=2)

    def get_vertices_callback(self):
        if len(self.polygons) == 0:
            pass
        else:
            for polygon in self.polygons:
                current_polygon_vertices = np.array([[point.x, point.y] for point in polygon.points])
                robot_is_in_area = self.is_point_inside_parallelogram(
                    self.robot_pose.pose.position.x,
                    self.robot_pose.pose.position.y,
                    current_polygon_vertices
                )
                if robot_is_in_area:
                    self.vertices = current_polygon_vertices
                    self.get_logger().info(
                        f'机器人在通道内，使用当前通道: \n{current_polygon_vertices}'
                    )
                    break


    def dis_point_to_point(self, p1_x, p1_y, p2_x, p2_y):
        return math.sqrt(math.pow(p1_x - p2_x, 2) + math.pow(p1_y - p2_y, 2))
    
    def dis_point_to_line(self, x, y, p1_x, p1_y, p2_x, p2_y):
        """
        计算点到线段的最短距离   改成是线段，不是无限长的直线
        """
        dx = p2_x - p1_x
        dy = p2_y - p1_y
        seg_len_sq = dx * dx + dy * dy

        # 线段退化成一个点
        if seg_len_sq == 0:
            return self.dis_point_to_point(x, y, p1_x, p1_y)

        # 计算垂足在线段上的投影参数 t
        t = ((x - p1_x) * dx + (y - p1_y) * dy) / seg_len_sq
        # 把 t 限制在 [0, 1]，保证垂足落在线段内
        t = max(0.0, min(1.0, t))

        # 垂足坐标
        proj_x = p1_x + t * dx
        proj_y = p1_y + t * dy

        return self.dis_point_to_point(x, y, proj_x, proj_y)
    

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
        self.get_logger().info('开始寻找避让点...')
        # self.get_logger().info(f'goal_handle.request..{goal_handle.request}')
        self.action_goal_handle_msg = goal_handle.request
        self.vehicle_width = self.action_goal_handle_msg.car_size.y
        # car_pose 有效性检查：frame_id 为空表示消息未填充
        car_pose = self.action_goal_handle_msg.car_pose
        
        # if car_pose.header.frame_id == '':
        #     self.get_logger().error('car_pose 为空/无效  frame_id 为空，abort')
        #     goal_handle.abort()
        #     return FindCarAvoidancePoint.Result()

        self.polygons = self.action_goal_handle_msg.polygons
        self.get_logger().info(f'polygons: {self.polygons}')
        
        # 发布所有通道多边形 marker
        self._publish_all_passages_marker(self.polygons)
        
        # 获取清洁区域信息
        # 每次需要用到self.vertices时，调用一下 get_vertices_callback()
        self.get_logger().info('寻找当前通行区域...')

        self.vertices = list(self.vertices)
        self.vertices.clear()
        self.get_vertices_callback()

        if len(self.vertices) == 0:
            self.get_logger().error('未找到用于寻找停靠点的通道')
            goal_handle.abort()
            return FindCarAvoidancePoint.Result()

        v1, v2, v3, v4 = self.vertices
        self.get_logger().info(
            f'当前通行区域: [({v1[0]:.3f}, {v1[1]:.3f}),({v2[0]:.3f}, {v2[1]:.3f}),'
            f'({v3[0]:.3f}, {v3[1]:.3f}),({v4[0]:.3f}, {v4[1]:.3f})]'
        )


        # self.get_logger().info(f'self.get_vertices_callback():{len(self.vertices)}')
        # self.get_logger().info(f'self.vertices:{self.vertices}')
        

        # 寻找停靠点
        self.get_logger().info('寻找停靠点...')
        avoidance_point = self.find_avoidance_point(self.robot_pose, self.vertices)
        if avoidance_point is not None:
            ap = avoidance_point.pose
            yaw = math.degrees(self.get_yaw_from_pose(avoidance_point))
            self.get_logger().info(
                f'成功找到避让点: x={ap.position.x:.3f}, y={ap.position.y:.3f}, '
                f'z={ap.position.z:.3f}, yaw={yaw:.3f}°'
            )

            # 发布最终避让点 marker
            nearest_boundary = self.find_nearest_boundary(self.robot_pose, self.vertices)
            car_pose = self.action_goal_handle_msg.car_pose.pose.position
            self._publish_debug_markers(
                self.robot_pose.pose.position.x,
                self.robot_pose.pose.position.y,
                car_pose,
                nearest_boundary,
                avoidance_point=avoidance_point
            )

            goal_handle.succeed()
            # goal_handle.abort()
            result = FindCarAvoidancePoint.Result()
            result.pose = avoidance_point
            
            rp = result.pose.pose
            yaw = math.degrees(self.get_yaw_from_pose(result.pose))
            self.get_logger().info(
                f'成功找到避让点*****: x={rp.position.x:.3f}, y={rp.position.y:.3f}, '
                f'z={rp.position.z:.3f}, yaw={yaw:.3f}°'
            )
            
            return result
        else:
            self.get_logger().error('外部停车点和自搜索均未找到有效避让点')
            goal_handle.abort()
            return FindCarAvoidancePoint.Result()

    def calculate_total_passage_width(self, vertices):

        # 计算每两点之间的长边
        side_length_list = [math.dist([vertices[i][0],vertices[i][1]],[vertices[(i + 1) % 4][0],vertices[(i + 1) % 4][1]]) for i in range(4)]
        distance = min(side_length_list)
        return distance
  
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
        
        # 线段ab的法向量
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
        
        # 计算向量ac在法向量上的投影
        normal_projection = ac_x * normal_x + ac_y * normal_y
        
        # 确定方向：始终向线段的另一侧移动
        direction = -1 if normal_projection > 0 else 1
        
        # 计算目标点坐标
        target_x = foot_x + direction * normal_x * distance
        target_y = foot_y + direction * normal_y * distance
        
        return (target_x, target_y)
    
    def _prepare_avoidance_search(self, robot_pose, cleaning_area_vertices):
        """
        准备避让点搜索所需的基础数据：
        - costmap 数组
        - 地图参数（origin, resolution, width, height）
        - nearest_boundary  最近长边
        - k   搜索方向角
        - k_diff2      k 与机器人朝向的夹角
        发布调试 marker。
        返回 None 表示准备失败。
        """
        if self.global_costmap is None:
            self.get_logger().error('全局代价地图未收到')
            return None

        # 地图数据
        map_info = self.global_costmap.metadata
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y
        resolution = map_info.resolution
        width = map_info.size_x
        height = map_info.size_y
        self.get_logger().info(f'origin_x: {origin_x:.3f}', once=True)
        self.get_logger().info(f'origin_y: {origin_y:.3f}', once=True)
        self.get_logger().info(f'resolution: {resolution:.3f}', once=True)
        self.get_logger().info(f'width: {width}', once=True)
        self.get_logger().info(f'height: {height}', once=True)

        map_data = np.array(self.global_costmap.data)
        costmap = np.ascontiguousarray(map_data.reshape((height, width)).astype(np.uint8))

        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y

        # 计算最近长边
        self.get_logger().info('寻找最近的边界...')
        nearest_boundary = self.find_nearest_boundary(robot_pose, cleaning_area_vertices)
        y_ = nearest_boundary[0][1] - nearest_boundary[1][1]
        x_ = nearest_boundary[0][0] - nearest_boundary[1][0]
        nb0 = nearest_boundary[0]
        nb1 = nearest_boundary[1]
        self.get_logger().info(
            f'nearest_boundary: [({nb0[0]:.3f}, {nb0[1]:.3f}), ({nb1[0]:.3f}, {nb1[1]:.3f})]'
        )

        # 计算搜索方向 k
        k = np.arctan2(y_, x_)
        self.get_logger().info(f'k_radian: {k:.3f}, k_degree: {k / math.pi * 180:.3f}')

        # 根据车的位置调整 k 方向
        car_pose = self.action_goal_handle_msg.car_pose.pose.position
        self.get_logger().info(f'robot: ({robot_x:.3f}, {robot_y:.3f})')
        self.get_logger().info(f'car: ({car_pose.x:.3f}, {car_pose.y:.3f})')

        # 统一发布基础调试 marker（车、机器人、最近长边）
        self._publish_debug_markers(robot_x, robot_y, car_pose, nearest_boundary)

        # 打印 footprint 等静态检查信息（只打一次）
        self._log_static_check_info_once(
            costmap, origin_x, origin_y, resolution, width, height
        )

        car_robot_k = np.arctan2(robot_y - car_pose.y, robot_x - car_pose.x)
        k_diff = self.angle_diff(k, car_robot_k)
        self.get_logger().info(f'k: {k:.3f}, car_robot_k: {car_robot_k:.3f}, k_diff: {k_diff:.3f}')
        if k_diff > math.pi / 2:
            k = self.add_angles(k, math.pi)

        # 计算 k 与机器人朝向的夹角
        orientation = robot_pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        k_robot = yaw
        k_diff2 = self.angle_diff(k, k_robot)
        self.get_logger().info(f'k: {k:.3f}, k_robot: {k_robot:.3f}, k_diff2: {k_diff2:.3f}')

        return {
            'costmap': costmap,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'resolution': resolution,
            'width': width,
            'height': height,
            'robot_x': robot_x,
            'robot_y': robot_y,
            'nearest_boundary': nearest_boundary,
            'k': k,
            'k_diff2': k_diff2,
            'car_pose': car_pose,
        }

    def _publish_debug_markers(self, robot_x, robot_y, car_pose, nearest_boundary,
                               find_vertices=None, avoidance_point=None):
        """
        统一发布所有调试用 marker
        """
        now = self.get_clock().now().to_msg()

        # 1. 车位置 marker
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = now
        msg.id = 1
        msg.type = Marker.CUBE
        msg.action = Marker.ADD
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = car_pose.x
        msg.pose.position.y = car_pose.y
        msg.pose.orientation.w = 1.0
        self.marker_car_pose_publisher.publish(msg)

        # 2. 机器人位置 marker
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = now
        msg.id = 2
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD
        msg.scale.x = 0.2
        msg.scale.y = 0.2
        msg.scale.z = 0.2
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = robot_x
        msg.pose.position.y = robot_y
        msg.pose.orientation.w = 1.0
        self.marker_robot_pose_publisher.publish(msg)

        # 3. 最近长边 marker
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = now
        msg.id = 3
        msg.type = Marker.LINE_LIST
        msg.action = Marker.ADD
        msg.scale.x = 0.2
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        p1 = Point()
        p1.x = nearest_boundary[0][0]
        p1.y = nearest_boundary[0][1]
        msg.points.append(p1)
        p2 = Point()
        p2.x = nearest_boundary[1][0]
        p2.y = nearest_boundary[1][1]
        msg.points.append(p2)
        self.marker_avoidance_side_publisher.publish(msg)

        # 4. 搜索矩形 marker
        if find_vertices is not None:
            msg = Marker()
            msg.header.frame_id = "map"
            msg.header.stamp = now
            msg.id = 4
            msg.type = Marker.LINE_LIST
            msg.action = Marker.ADD
            msg.scale.x = 0.1
            msg.color.r = 1.0
            msg.color.g = 0.0
            msg.color.b = 0.0
            msg.color.a = 1.0
            size_tmp = len(find_vertices)
            for i in range(size_tmp):
                p_start = Point()
                p_start.x = find_vertices[i][0]
                p_start.y = find_vertices[i][1]
                msg.points.append(p_start)
                p_end = Point()
                p_end.x = find_vertices[(i + 1) % size_tmp][0]
                p_end.y = find_vertices[(i + 1) % size_tmp][1]
                msg.points.append(p_end)
            self.marker_searching_rect_publisher.publish(msg)

        # 5. 避让点 marker
        if avoidance_point is not None:
            msg = Marker()
            msg.header.frame_id = "map"
            msg.header.stamp = now
            msg.id = 5
            msg.type = Marker.ARROW
            msg.action = Marker.ADD
            msg.scale.x = 0.5
            msg.scale.y = 0.2
            msg.scale.z = 0.4
            msg.color.r = 0.0
            msg.color.g = 0.0
            msg.color.b = 1.0
            msg.color.a = 1.0
            msg.pose = avoidance_point.pose
            self.marker_parking_point_publisher.publish(msg)

    # 寻找所有的避让点
    def find_avoidance_point(self, robot_pose, cleaning_area_vertices):
        prep = self._prepare_avoidance_search(robot_pose, cleaning_area_vertices)
        if prep is None:
            return None

        costmap = prep['costmap']
        origin_x = prep['origin_x']
        origin_y = prep['origin_y']
        resolution = prep['resolution']
        width = prep['width']
        height = prep['height']
        robot_x = prep['robot_x']
        robot_y = prep['robot_y']
        nearest_boundary = prep['nearest_boundary']
        k = prep['k']
        k_diff2 = prep['k_diff2']
        car_pose = prep['car_pose']

        # ===== 优先尝试外部停车点 =====
        car_pose = self.action_goal_handle_msg.car_pose.pose.position
        external_candidates = self._filter_external_stop_points(
            robot_x, robot_y, car_pose.x, car_pose.y, k
        )   # 把超过8m的点都删除了
        if len(external_candidates) > 0:
            self.get_logger().info(f'尝试外部停车点，共{len(external_candidates)}个')
            valid_external = []    # 所有的满足的外部停车点
            for ext_pose in external_candidates:
                ext_is_inside = self.is_point_inside_parallelogram(
                    ext_pose.pose.position.x, ext_pose.pose.position.y, self.vertices
                )
                if ext_is_inside:
                    # 通道内：先按避让方向覆盖 yaw，再校验（与自搜索点一致）
                    final_yaw = self._compute_avoidance_yaw(
                        k, robot_x, robot_y, car_pose.x, car_pose.y
                    )
                    quat = Quaternion()
                    quat.x, quat.y, quat.z, quat.w = quaternion_from_euler(0, 0, final_yaw)
                    ext_pose.pose.orientation = quat
                else:
                    self.get_logger().info(
                        f'外部停车点({ext_pose.pose.position.x:.2f}, '
                        f'{ext_pose.pose.position.y:.2f}) 在通道外，将跳过服务检查'
                    )

                skip_service = not ext_is_inside

                # 对单个点进行完整的筛选流程
                if self._validate_candidate(
                    ext_pose, costmap, robot_x, robot_y,
                    origin_x, origin_y, resolution, width, height, nearest_boundary,
                    skip_service_check=skip_service
                ):
                    valid_external.append(ext_pose)

            if len(valid_external) > 0:
                # 新增的一个   选离 nearest_boundary  机器人最近的长边最近的点   
                best = min(valid_external, key=lambda pose: self.dis_point_to_line(
                    pose.pose.position.x, pose.pose.position.y,
                    nearest_boundary[0][0], nearest_boundary[0][1],
                    nearest_boundary[1][0], nearest_boundary[1][1]
                ))
                # 判断外部停靠点是否在通道内，并设置对应的 z 值与方向
                is_inside = self.is_point_inside_parallelogram(
                    best.pose.position.x, best.pose.position.y, self.vertices
                )
                if is_inside:
                    best.pose.position.z = 20.0
                    self.get_logger().info(
                        f'外部停车点通过校验共{len(valid_external)}个，'
                        f'选择最贴近长边: ({best.pose.position.x:.2f}, {best.pose.position.y:.2f})，'
                        f'该点在通道内，使用避让方向，z = 20.0'
                    )
                else:
                    # 通道外的固定停靠点：方向值保持不变
                    best.pose.position.z = 30.0
                    self.get_logger().info(
                        f'外部停车点通过校验共{len(valid_external)}个，'
                        f'选择最贴近长边: ({best.pose.position.x:.2f}, {best.pose.position.y:.2f})，'
                        f'该点在通道外，方向保持不变，z = 30.0'
                    )
                return best

            self.get_logger().info('有外部停车点，但是所有外部停车点在筛选判断以后均不满足，回退自搜索')
        else:
            self.get_logger().info('话题无有效外部停车点，使用自搜索')
        # ===== 外部点逻辑结束，继续原有搜索矩形逻辑 =====

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

        # 计算通道宽度和终止条件
        passage_width = self.calculate_total_passage_width(self.vertices)
        half_width = passage_width / 2.0
        self.get_logger().info(f'通道宽度: {passage_width:.3f}, 半宽: {half_width:.3f}')

        # 单次搜索：只在当前偏移位置搜索一次
        avoidance_pose_result = None

        offset_min = self.outside_min
        offset_max = self.outside_max

        self.get_logger().info(
            f'当前搜索偏移: offset_min={offset_min:.2f}, offset_max={offset_max:.2f}'
        )

        # 根据机器人是否在通道内部，决定偏移方向
        if self.is_point_inside_parallelogram(robot_x, robot_y, self.vertices):
            # 机器人在通道内部
            p1_near = self.findIntersection(
                nearest_boundary[0], nearest_boundary[1],
                [robot_x1, robot_y1], offset_min
            )
            p2_near = self.findIntersection(
                nearest_boundary[0], nearest_boundary[1],
                [robot_x2, robot_y2], offset_min
            )
            p1_far = self.findIntersection(
                nearest_boundary[0], nearest_boundary[1],
                [robot_x1, robot_y1], offset_max
            )
            p2_far = self.findIntersection(
                nearest_boundary[0], nearest_boundary[1],
                [robot_x2, robot_y2], offset_max
            )
        else:
            # 机器人在通道外部
            # 复用 dis_point_to_line 找最近边，并记录它的长度
            verts = self.vertices
            min_dis = float('inf')
            nearest_edge_len = 0.0
            for i in range(4):
                p1 = verts[i]
                p2 = verts[(i + 1) % 4]
                d = self.dis_point_to_line(robot_x, robot_y, p1[0], p1[1], p2[0], p2[1])
                if d < min_dis:
                    min_dis = d
                    nearest_edge_len = self.dis_point_to_point(p1[0], p1[1], p2[0], p2[1])

            short_edge_len = self.calculate_total_passage_width(verts)  # 复用：最短边长度
            # 最近边长度接近短边长度 → 从短边出去；明显更长 → 从长边出去
            nearest_is_long = nearest_edge_len > short_edge_len * 1.5

            dis_robot_to_nearest_bound = min_dis  # 复用这个值，下面 offset_min_local 也用它

            offset_min_local = offset_min
            if offset_min < dis_robot_to_nearest_bound < offset_max:
                offset_min_local = dis_robot_to_nearest_bound
            elif nearest_is_long and dis_robot_to_nearest_bound >= self.stop_in_place_min_dist:
                # 从长边出去、且离边够远 → 路肩位置，可原地停
                self.get_logger().info(
                    f"机器人在长边外侧 {dis_robot_to_nearest_bound:.2f}m，原地停车"
                )
                ret_pose = PoseStamped()
                ret_pose.header.stamp = self.get_clock().now().to_msg()
                ret_pose.header.frame_id = "map"
                ret_pose.pose.position.x = robot_x
                ret_pose.pose.position.y = robot_y
                target_angle = math.degrees(math.atan2(
                    nearest_boundary[1][1] - nearest_boundary[0][1],
                    nearest_boundary[1][0] - nearest_boundary[0][0]
                ))
                yaw = self.get_yaw_from_pose(robot_pose)
                ret_pose_yaw = math.radians(self.adjust_angle((math.cos(yaw), math.sin(yaw)), target_angle))
                quat = Quaternion()
                quat.x, quat.y, quat.z, quat.w = quaternion_from_euler(0, 0, ret_pose_yaw)
                ret_pose.pose.orientation = quat
                ret_pose.pose.position.z = 10.0
                return ret_pose
            else:
                self.get_logger().info(
                    f"出口边是{'长边' if nearest_is_long else '短边'}, 距离{dis_robot_to_nearest_bound:.2f}m，"
                    f"不满足原地停条件，继续搜索"
                )

            p1_near = self.findIntersection(
                nearest_boundary[0], nearest_boundary[1],
                [robot_x1, robot_y1], -offset_min_local
            )
            p2_near = self.findIntersection(
                nearest_boundary[0], nearest_boundary[1],
                [robot_x2, robot_y2], -offset_min_local
            )
            p1_far = self.findIntersection(
                nearest_boundary[0], nearest_boundary[1],
                [robot_x1, robot_y1], -offset_max
            )
            p2_far = self.findIntersection(
                nearest_boundary[0], nearest_boundary[1],
                [robot_x2, robot_y2], -offset_max
            )

        if p1_near is None or p2_near is None or p1_far is None or p2_far is None:
            self.get_logger().info('findIntersection 返回 None，放弃搜索')
            return None

        # 搜索框最前面的短边不能越过通道前方短边
        forward_short_edge = self.get_forward_short_edge(robot_x, robot_y, self.vertices, k)
        polygon_center = np.mean(np.array(self.vertices), axis=0)
        if self.is_point_beyond_edge(
            [p1_near[0], p1_near[1]],
            forward_short_edge,
            polygon_center
        ) or self.is_point_beyond_edge(
            [p1_far[0], p1_far[1]],
            forward_short_edge,
            polygon_center
        ):
            self.get_logger().error(
                '搜索框构造失败：搜索框最前面的短边越过了通道前方短边'
            )
            return None

        find_vertices = self.sort_quadrilateral_vertices([
            p1_near, p1_far, p2_far, p2_near
        ])

        fv = find_vertices
        self.get_logger().info(
            f'find_vertices: [({fv[0][0]:.3f}, {fv[0][1]:.3f}),({fv[1][0]:.3f}, {fv[1][1]:.3f}),'
            f'({fv[2][0]:.3f}, {fv[2][1]:.3f}),({fv[3][0]:.3f}, {fv[3][1]:.3f})]'
        )

        # 发布搜索区域 marker
        self._publish_debug_markers(
            robot_x, robot_y, car_pose, nearest_boundary,
            find_vertices=find_vertices
        )

        # 在当前搜索区域内找避让点
        search_posestamped_list = self.select_points_in_parallelogram(find_vertices, self.search_point_interval, k)
        self.get_logger().info(f'search_posestamped length: {len(search_posestamped_list)}')

        if len(search_posestamped_list) > 0:
            self.get_logger().info(f'一共{len(search_posestamped_list)}个候选避让点')

            for avoidance_pose in search_posestamped_list:
                if self._validate_candidate(
                    avoidance_pose, costmap, robot_x, robot_y,
                    origin_x, origin_y, resolution, width, height, nearest_boundary
                ):
                    avoidance_pose_result = avoidance_pose
                    avoidance_pose_result.pose.position.z = 10.0
                    return avoidance_pose_result

        # 单次搜索结束，没找到
        self.get_logger().info('当前搜索区域没有找到避让点')
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

    def _publish_all_passages_marker(self, polygons):
        """
        发布所有通道多边形到 marker_all_passages，用蓝色细线可视化。
        :param polygons: list of garage_utils_msgs.msg.Polygon
        """
        now = self.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = now
        marker.id = 20
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.8

        if not polygons or len(polygons) == 0:
            marker.action = Marker.DELETE
            self.marker_all_passages_publisher.publish(marker)
            return

        for polygon in polygons:
            pts = polygon.points
            if len(pts) < 2:
                continue
            n = len(pts)
            for i in range(n):
                p1 = pts[i]
                p2 = pts[(i + 1) % n]
                point1 = Point()
                point1.x = p1.x
                point1.y = p1.y
                point1.z = p1.z
                marker.points.append(point1)
                point2 = Point()
                point2.x = p2.x
                point2.y = p2.y
                point2.z = p2.z
                marker.points.append(point2)

        self.marker_all_passages_publisher.publish(marker)

    def _special_terrain_callback(self, msg):
        self.special_terrain_polygons = msg.polygons
        for idx, polygon in enumerate(msg.polygons):
            pts_str = ', '.join([f'({p.x:.3f}, {p.y:.3f})' for p in polygon.points])
            self.get_logger().info(
                f'禁扫区[{idx}]: {pts_str}',
                throttle_duration_sec=20.0
            )
        self.get_logger().info(
            f'收到禁扫区: {len(msg.polygons)}个区域', once=True
        )

        # 发布禁扫区 Marker 用于 rviz2 可视化
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 10
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.05

        if len(msg.polygons) == 0:
            marker.action = Marker.DELETE
            self.marker_special_terrain_publisher.publish(marker)
            return

        marker.action = Marker.ADD
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        for polygon in msg.polygons:
            pts = polygon.points
            if len(pts) < 2:
                continue
            n = len(pts)
            for i in range(n):
                p1 = pts[i]
                p2 = pts[(i + 1) % n]
                point1 = Point()
                point1.x = p1.x
                point1.y = p1.y
                point1.z = p1.z
                marker.points.append(point1)
                point2 = Point()
                point2.x = p2.x
                point2.y = p2.y
                point2.z = p2.z
                marker.points.append(point2)

        self.marker_special_terrain_publisher.publish(marker)

    def _vehicle_stop_points_callback(self, msg):
        self.vehicle_avoidance_stop_points = msg
        self.get_logger().info(
            f'收到避车停车点: {len(msg.poses)}个',
            throttle_duration_sec=5.0
        )

        if len(msg.poses) > 0:
            orientation = msg.poses[0].orientation
            is_default_orientation = (
                abs(orientation.x) < 1e-6 and
                abs(orientation.y) < 1e-6 and
                abs(orientation.z) < 1e-6 and
                abs(abs(orientation.w) - 1.0) < 1e-6
            )
            if is_default_orientation:
                self.get_logger().info(
                    '外部停车点没有发 yaw，将使用避让方向自行计算',
                    throttle_duration_sec=5.0
                )

    def _filter_external_stop_points(self, robot_x, robot_y, car_x, car_y, k):
        """
        过滤外部避车停车点：
        """
        if self.vehicle_avoidance_stop_points is None:
            return []
        if len(self.vehicle_avoidance_stop_points.poses) == 0:
            return []

        # 从机器人当前位姿提取 yaw
        robot_yaw = self.get_yaw_from_pose(self.robot_pose)
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        def to_base_link(wx, wy):
            dx = wx - robot_x
            dy = wy - robot_y
            local_x = dx * cos_yaw + dy * sin_yaw
            local_y = -dx * sin_yaw + dy * cos_yaw
            return local_x, local_y

        car_local_x, _ = to_base_link(car_x, car_y)

        results = []
        for pose in self.vehicle_avoidance_stop_points.poses:
            px = pose.position.x
            py = pose.position.y

            # 距离过滤
            dist = math.sqrt((px - robot_x) ** 2 + (py - robot_y) ** 2)
            if dist > self.external_point_max_distance:
                continue

            # base_link 与车在 x 轴异侧才保留
            point_local_x, _ = to_base_link(px, py)
            if car_local_x * point_local_x >= 0:
                self.get_logger().info(
                    f'外部停车点 ({px:.2f}, {py:.2f}) 与车同侧，过滤掉'
                )
                continue

            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = px
            pose_stamped.pose.position.y = py
            pose_stamped.pose.position.z = 0.0

            # 先保留外部点的原始方向，通道内/外的方向处理放到 find_avoidance_point 中决定
            pose_stamped.pose.orientation = pose.orientation

            results.append((dist, pose_stamped))

        # 按距离升序
        results.sort(key=lambda x: x[0])
        self.get_logger().info(f'外部停车点过滤后剩余: {len(results)}个')
        return [ps for _, ps in results]

    def _validate_candidate(self, avoidance_pose, costmap, robot_x, robot_y,
                             origin_x, origin_y, resolution, width, height, nearest_boundary,
                             skip_service_check=False):
        """
        对单个候选点执行完整校验链
        返回 True 通过，False 不通过
        """
        px_point = avoidance_pose.pose.position.x
        py_point = avoidance_pose.pose.position.y

        # ===== 位置硬约束：车和点必须分居机器人前后两侧 =====
        car_x = self.action_goal_handle_msg.car_pose.pose.position.x
        car_y = self.action_goal_handle_msg.car_pose.pose.position.y
        robot_yaw = self.get_yaw_from_pose(self.robot_pose)

        # 车和点都转到 base_link，看局部 x（前为正，后为负）
        def _local_x(wx, wy):
            dx = wx - robot_x
            dy = wy - robot_y
            return dx * math.cos(robot_yaw) + dy * math.sin(robot_yaw)

        car_local_x = _local_x(car_x, car_y)
        point_local_x = _local_x(px_point, py_point)

        # 一前一后 → 乘积为负 → 合格；同侧 → 乘积>=0 → 拒绝
        if car_local_x * point_local_x >= 0:
            self.get_logger().info(
                f'候选点({px_point:.2f},{py_point:.2f}) 与车在机器人同侧'
                f'(car_local_x={car_local_x:.2f}, point_local_x={point_local_x:.2f})，拒绝'
            )
            return False

        # 像素坐标
        point_pixel = (np.array([px_point, py_point]) - np.array([origin_x, origin_y])) / resolution
        point_pixel[0] = np.clip(point_pixel[0], 0, width - 1)
        point_pixel[1] = np.clip(point_pixel[1], 0, height - 1)
        point_x_p = int(point_pixel[0])
        point_y_p = int(point_pixel[1])

        # 3-1 点周围无障碍
        point_free_radius_pixels = max(1, int(self.point_free_check_radius / resolution))
        if not self.check_point_is_free(costmap, (point_x_p, point_y_p), radius=point_free_radius_pixels):
            self.get_logger().info(
                f'候选点({px_point:.2f},{py_point:.2f}) 周围{self.point_free_check_radius}m'
                f'（{point_free_radius_pixels}像素）内有障碍'
            )
            return False

        # 3-2 不在禁扫区
        if self._is_point_in_special_terrain(px_point, py_point):
            self.get_logger().info(f'候选点({px_point:.2f},{py_point:.2f}) 在禁扫区内')
            return False

        # 3-3 footprint不与禁扫区重叠
        pose_yaw = euler_from_quaternion([
            avoidance_pose.pose.orientation.x,
            avoidance_pose.pose.orientation.y,
            avoidance_pose.pose.orientation.z,
            avoidance_pose.pose.orientation.w
        ])[2]
        if self._is_footprint_in_special_terrain(px_point, py_point, pose_yaw):
            self.get_logger().info(f'候选点({px_point:.2f},{py_point:.2f}) footprint与禁扫区重叠')
            return False

        # 3-4 连线无障碍
        robot_pixel = (np.array([robot_x, robot_y]) - np.array([origin_x, origin_y])) / resolution
        robot_pixel[0] = np.clip(robot_pixel[0], 0, width - 1)
        robot_pixel[1] = np.clip(robot_pixel[1], 0, height - 1)
        robot_x_p = int(robot_pixel[0])
        robot_y_p = int(robot_pixel[1])

        if not self.check_path_is_free(costmap, (robot_x_p, robot_y_p), (point_x_p, point_y_p)):
            self.get_logger().info(f'候选点({px_point:.2f},{py_point:.2f}) 连线有障碍')
            return False

        # 3-5 footprint覆盖安全
        if not self.check_footprint_at_pose(
            costmap, px_point, py_point, pose_yaw,
            origin_x, origin_y, resolution, width, height
        ):
            self.get_logger().info(f'候选点({px_point:.2f},{py_point:.2f}) footprint撞障碍')
            return False

        # 3-6 扫掠矩形安全
        if not self.check_footprint_sweep(
            costmap, self.robot_pose, avoidance_pose, pose_yaw,
            self.footprint_sweep_long_step,
            self.footprint_sweep_short_step,
            origin_x, origin_y, resolution, width, height
        ):
            self.get_logger().info(f'候选点({px_point:.2f},{py_point:.2f}) 扫掠区域有障碍')
            return False

        # ===== 外部停车点且在通道外：跳过服务检查，直接通过 =====
        if skip_service_check:
            self.get_logger().info(
                f'候选点({px_point:.2f},{py_point:.2f}) 为通道外外部停车点，'
                f'跳过 /check_car_passable 服务检查，直接通过'
            )
            return True

        # 服务检查
        avoidance_pose_msg = IsCarPassable.Request()
        avoidance_pose_msg.robot_pose = avoidance_pose
        avoidance_pose_msg.car_pose = self.action_goal_handle_msg.car_pose
        avoidance_pose_msg.size = self.action_goal_handle_msg.car_size
        start_time = time.time()
        check_result = self.check_avoidance(avoidance_pose_msg)
        end_time = time.time()
        self.get_logger().info(f'服务检查 result={check_result}, time={end_time - start_time:.3f}s')

        if check_result and (end_time - start_time) < self.check_service_max_time:
            return True

        return False

    def _is_point_in_special_terrain(self, x, y):
        """检查点(x, y)是否在任何禁扫区多边形内"""
        if self.special_terrain_polygons is None:
            return False
        for polygon in self.special_terrain_polygons:
            if not polygon.points or len(polygon.points) < 3:
                continue
            poly_pts = np.array(
                [[p.x, p.y] for p in polygon.points], dtype=np.float32
            )
            result = cv2.pointPolygonTest(poly_pts, (float(x), float(y)), False)
            if result >= 0:
                return True
        return False

    def _is_footprint_in_special_terrain(self, x, y, yaw):
        """检查机器人在(x, y, yaw)处时，footprint是否与任何禁扫区多边形重叠"""
        if self.special_terrain_polygons is None or not self.footprint_vertices:
            return False

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 计算 footprint 在世界坐标系下的顶点
        fp_world = []
        for vx, vy in self.footprint_vertices:
            wx = x + vx * cos_yaw - vy * sin_yaw
            wy = y + vx * sin_yaw + vy * cos_yaw
            fp_world.append([wx, wy])
        fp_contour = np.array(fp_world, dtype=np.float32)

        for polygon in self.special_terrain_polygons:
            if not polygon.points or len(polygon.points) < 3:
                continue
            poly_pts = np.array(
                [[p.x, p.y] for p in polygon.points], dtype=np.float32
            )

            # 检查1: footprint 任意顶点在禁扫区内
            for pt in fp_world:
                if cv2.pointPolygonTest(poly_pts, (pt[0], pt[1]), False) >= 0:
                    return True

            # 检查2: 禁扫区任意顶点在 footprint 内
            for pt in poly_pts:
                if cv2.pointPolygonTest(fp_contour, (float(pt[0]), float(pt[1])), False) >= 0:
                    return True

            # 检查3: 边是否相交   用 cv2 计算交集面积
            ret, intersection = cv2.intersectConvexConvex(fp_contour, poly_pts)
            if ret > 0:
                return True

        return False

    # 判断机器人是否在某个边框内
    def is_point_inside_parallelogram(self, robot_x, robot_y, vertices):
        """
        使用射线法判断点是否在多边形内部
        """
        inside = False
        j = len(vertices) - 1
        
        for i in range(len(vertices)):
            xi, yi = vertices[i][0], vertices[i][1]
            xj, yj = vertices[j][0], vertices[j][1]
            
            # 判断边是否与从(robot_x, robot_y)出发的水平射线相交
            intersect = ((yi > robot_y) != (yj > robot_y)) and \
                        (robot_x < (xj - xi) * (robot_y - yi) / (yj - yi) + xi)
            
            if intersect:
                inside = not inside
            j = i  # 更新j为当前i，用于下一次迭代
        
        return inside
    
    # 选择平行四边形区域内的点
    def select_points_in_parallelogram(self, vertices, interval, direction):
        generate_search_points_without_directions = self.generate_all_search_points(vertices, interval)
        generate_search_points_with_directions = self.process_points(self.robot_pose, vertices, generate_search_points_without_directions, direction)
        search_posetampd_list = []
        for point_with_direction in generate_search_points_with_directions:
            pose_with_direction = PoseStamped()
            pose_with_direction.header.stamp = self.get_clock().now().to_msg()
            pose_with_direction.header.frame_id = 'map'
            pose_with_direction.pose.position.x = point_with_direction[0][0]
            pose_with_direction.pose.position.y = point_with_direction[0][1]
            quat = Quaternion()
            quat.x, quat.y, quat.z, quat.w = quaternion_from_euler(0, 0, math.radians(point_with_direction[1]))
            pose_with_direction.pose.orientation = quat
            search_posetampd_list.append(pose_with_direction)
        # self.show(generate_search_points_with_directions)
        return search_posetampd_list
    
    def generate_all_search_points(self,vertices, interval):
        A = np.array(vertices[0])
        B = np.array(vertices[1])
        D = np.array(vertices[3])
        
        u = B - A  
        v = D - A  
        
        u_length = np.linalg.norm(u)
        v_length = np.linalg.norm(v)
        steps_u = int(u_length / interval)
        steps_v = int(v_length / interval)
        
        grid_points = []
        grid_points.append(tuple(np.round(A, 6)))

        for i in range(steps_u + 1):
            for j in range(steps_v + 1):
                if i == 0 and j == 0:
                    continue
                offset_u = interval * i
                offset_v = interval * j
                if offset_u > u_length or offset_v > v_length:
                    continue
                point = A + (u / u_length) * offset_u + (v / v_length) * offset_v
                grid_points.append(tuple(np.round(point, 6)))
        
        
        generate_points = sorted(grid_points, key=self.distance_sq)
        return generate_points
    
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
    
    def _compute_avoidance_yaw(self, k, robot_x, robot_y, car_x, car_y):
        """沿通道长边方向，选择与来车方向背对的 yaw（与自搜索点一致）"""
        to_car_x = car_x - robot_x
        to_car_y = car_y - robot_y
        k_vec = (math.cos(k), math.sin(k))
        dot = k_vec[0] * to_car_x + k_vec[1] * to_car_y
        return k if dot < 0 else k + math.pi

    def process_points(self, robot_pose, vertices, points, direction):
        car_x = self.action_goal_handle_msg.car_pose.pose.position.x
        car_y = self.action_goal_handle_msg.car_pose.pose.position.y
        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y

        final_yaw = self._compute_avoidance_yaw(
            direction, robot_x, robot_y, car_x, car_y
        )
        final_angle = math.degrees(final_yaw)

        results = []
        for point in points:
            results.append((point, final_angle))

        return results
    

    # 对四边形的四个顶点进行排序，按左上角、右上角、右下角、左下角的顺序返回
    def sort_quadrilateral_vertices(self, points):
        points = np.array(points)
        
        # 1. 计算中心点
        center = np.mean(points, axis=0)
        
        # 2. 计算每个点相对于中心的角度
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        
        # 3. 按角度排序
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        # 4. 确保顺序是 [左上, 右上, 右下, 左下]
        # 找到 y 值最小的两个点
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

        A, B, C = line_coeffs
        x1, y1 = a

        # 计算对称点的公式
        denominator = A ** 2 + B ** 2
        x2 = x1 - 2 * A * (A * x1 + B * y1 + C) / denominator
        y2 = y1 - 2 * B * (A * x1 + B * y1 + C) / denominator

        return (x2, y2)
    
    def _log_static_check_info_once(self, costmap, origin_x, origin_y, resolution, width, height):
        """
        footprint 大小和地图分辨率都是静态的，只在第一次拿到两者后，
        打印一次 footprint 覆盖栅格数等固定信息。
        """
        if self._static_check_info_logged or not self.footprint_vertices:
            return

        # 以 yaw=0 计算 footprint 覆盖栅格数（仅用于日志，不影响实际检查）
        pixel_vertices = []
        for vx, vy in self.footprint_vertices:
            px = int((vx - origin_x) / resolution)
            py = int((vy - origin_y) / resolution)
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))
            pixel_vertices.append([px, py])

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(pixel_vertices, dtype=np.int32), 255)
        total_points = int(np.count_nonzero(mask == 255))

        fp_str = ', '.join([f'({v[0]:.3f}, {v[1]:.3f})' for v in self.footprint_vertices])
        self.get_logger().info(
            f'获取当前格子数量和地图的边长 footprint 覆盖栅格数={total_points}, '
            f'分辨率={resolution:.3f}m, footprint_vertices=[{fp_str}]'
        )
        self._static_check_info_logged = True

    # 检查点附近是否有障碍物
    def check_point_is_free(self, image, center, radius=2):
        """
        判断以指定像素点为中心、半径为radius 像素的圆形区域内所有像素值是否都等于0
        :param image: 输入的单通道图像（灰度图）
        :param center: 中心像素点的坐标 (x, y)
        :param radius: 圆形区域的半径
        :return: 如果圆形区域内所有像素值都等于 0 返回 True，否则返回 False
        """
        height, width = image.shape

        # 遍历圆形区域内的所有像素
        for y in range(max(0, int(center[1] - radius)), min(height, int(center[1] + radius + 1))):
            for x in range(max(0, int(center[0] - radius)), min(width, int(center[0] + radius + 1))):
                # 计算当前像素到中心像素的距离
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                if distance <= radius:
                    # 检查像素值是否等于 0
                    if image[y, x] == 254:
                        return False
        return True

    def check_footprint_at_pose(self, costmap, x, y, yaw, origin_x, origin_y, resolution, width, height):
        """
        检查机器人在给定位姿时，footprint 覆盖的所有栅格是否触碰障碍物
        :return: 全部安全返回 True，否则返回 False
        """
        if not self.footprint_vertices:
            return True

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 第一步：4个顶点旋转平移，转像素坐标
        pixel_vertices = []
        for vx, vy in self.footprint_vertices:
            rx = vx * cos_yaw - vy * sin_yaw
            ry = vx * sin_yaw + vy * cos_yaw
            wx = x + rx
            wy = y + ry
            px = int((wx - origin_x) / resolution)
            py = int((wy - origin_y) / resolution)
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))
            pixel_vertices.append([px, py])

        # 第二步：填充多边形，获取footprint覆盖的所有栅格
        pixel_vertices_np = np.array(pixel_vertices, dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pixel_vertices_np, 255)

        # 第三步：检查覆盖区域内costmap值
        ys, xs = np.where(mask == 255)
        total_points = len(xs)
        values = costmap[ys, xs]
        is_safe = bool(np.all(values <= 253))

        if not is_safe:
            self.get_logger().warning(
                f'check_footprint_at_pose: footprint 撞障碍 '
                f'({total_points} 个栅格点)'
            )

        return is_safe

    def check_footprint_sweep(self, costmap, robot_pose, target_pose, target_yaw,
                              long_edge_step, short_edge_step,
                              origin_x, origin_y, resolution, width, height, threshold=253):
        """
        检查机器人到候选点的扫掠矩形区域是否安全
        """
        if not self.footprint_vertices:
            return True

        # 计算 footprint 边长
        edge_lengths = []
        for i in range(len(self.footprint_vertices)):
            v1 = np.array(self.footprint_vertices[i])
            v2 = np.array(self.footprint_vertices[(i + 1) % len(self.footprint_vertices)])
            edge_lengths.append(np.linalg.norm(v2 - v1))
        footprint_long_edge = max(edge_lengths)
        footprint_short_edge = min(edge_lengths)

        # 如果 long_step 大于 footprint 长边，不启用
        if long_edge_step > footprint_long_edge:
            return True

        #  生成两个 footprint 的世界坐标角点
        cos_yaw = math.cos(target_yaw)
        sin_yaw = math.sin(target_yaw)

        robot_vertices = []
        for vx, vy in self.footprint_vertices:
            rx = vx * cos_yaw - vy * sin_yaw
            ry = vx * sin_yaw + vy * cos_yaw
            wx = robot_pose.pose.position.x + rx
            wy = robot_pose.pose.position.y + ry
            robot_vertices.append((wx, wy))

        target_vertices = []
        for vx, vy in self.footprint_vertices:
            rx = vx * cos_yaw - vy * sin_yaw
            ry = vx * sin_yaw + vy * cos_yaw
            wx = target_pose.pose.position.x + rx
            wy = target_pose.pose.position.y + ry
            target_vertices.append((wx, wy))

        # 找到最远的两条不重复连线
        best_pair = None
        best_total_distance = 0

        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        if i == k or j == l:
                            continue
                        dist1 = math.sqrt(
                            (target_vertices[j][0] - robot_vertices[i][0]) ** 2 +
                            (target_vertices[j][1] - robot_vertices[i][1]) ** 2
                        )
                        dist2 = math.sqrt(
                            (target_vertices[l][0] - robot_vertices[k][0]) ** 2 +
                            (target_vertices[l][1] - robot_vertices[k][1]) ** 2
                        )
                        total = dist1 + dist2
                        if total > best_total_distance:
                            best_total_distance = total
                            best_pair = (
                                (robot_vertices[i], target_vertices[j]),
                                (robot_vertices[k], target_vertices[l])
                            )

        if best_pair is None:
            return True

        # X 的四个端点就是扫掠矩形的四个角点
        sweep_vertices = [
            best_pair[0][0],  # 机器人角点1
            best_pair[0][1],  # 候选点角点1
            best_pair[1][1],  # 候选点角点2
            best_pair[1][0],  # 机器人角点2
        ]

        # 保证顶点按凸多边形顺序排列，cv2.fillConvexPoly 需要凸多边形顶点顺序
        sweep_vertices = self.sort_quadrilateral_vertices(sweep_vertices)

        # 建立局部坐标系
        dx = target_pose.pose.position.x - robot_pose.pose.position.x
        dy = target_pose.pose.position.y - robot_pose.pose.position.y
        move_distance = math.sqrt(dx ** 2 + dy ** 2)

        if move_distance < 1e-6:
            return True

        long_axis_x = dx / move_distance
        long_axis_y = dy / move_distance
        short_axis_x = -long_axis_y
        short_axis_y = long_axis_x

        #  将四个角点投影到局部坐标系，找到采样范围
        local_coords = []
        for wx, wy in sweep_vertices:
            offset_x = wx - robot_pose.pose.position.x
            offset_y = wy - robot_pose.pose.position.y
            local_long = offset_x * long_axis_x + offset_y * long_axis_y
            local_short = offset_x * short_axis_x + offset_y * short_axis_y
            local_coords.append((local_long, local_short))

        long_min = min(c[0] for c in local_coords)
        long_max = max(c[0] for c in local_coords)
        short_min = min(c[1] for c in local_coords)
        short_max = max(c[1] for c in local_coords)

        # 任一步长为 0，检查所有点
        if short_edge_step == 0 or long_edge_step == 0:
            pixel_vertices = []
            for wx, wy in sweep_vertices:
                px = int((wx - origin_x) / resolution)
                py = int((wy - origin_y) / resolution)
                px = max(0, min(width - 1, px))
                py = max(0, min(height - 1, py))
                pixel_vertices.append([px, py])

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.array(pixel_vertices, dtype=np.int32), 255)
            ys, xs = np.where(mask == 255)
            if len(xs) == 0:
                return True
            values = costmap[ys, xs]
            is_safe = bool(np.all(values <= threshold))
            if not is_safe:
                self.get_logger().warning(
                    f'check_footprint_sweep (filled): {len(xs)} pixels 中有障碍'
                )
            return is_safe

        # short_step > 短边，只检查两条长边
        if short_edge_step > footprint_short_edge:
            pixel_vertices = []
            for wx, wy in sweep_vertices:
                px = int((wx - origin_x) / resolution)
                py = int((wy - origin_y) / resolution)
                px = max(0, min(width - 1, px))
                py = max(0, min(height - 1, py))
                pixel_vertices.append((px, py))

            # 识别两条长边
            edge_lengths_px = []
            for i in range(4):
                p1 = pixel_vertices[i]
                p2 = pixel_vertices[(i + 1) % 4]
                length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                edge_lengths_px.append((i, length))
            sorted_edges = sorted(edge_lengths_px, key=lambda x: -x[1])
            long_edge_1_idx = sorted_edges[0][0]
            long_edge_2_idx = sorted_edges[1][0]

            # bresenham 提取两条长边的像素
            p1 = pixel_vertices[long_edge_1_idx]
            p2 = pixel_vertices[(long_edge_1_idx + 1) % 4]
            edge1_points = self.bresenham(p1[0], p1[1], p2[0], p2[1], costmap)

            p3 = pixel_vertices[long_edge_2_idx]
            p4 = pixel_vertices[(long_edge_2_idx + 1) % 4]
            edge2_points = self.bresenham(p3[0], p3[1], p4[0], p4[1], costmap)

            all_points = np.vstack([edge1_points, edge2_points])
            values = costmap[all_points[:, 1], all_points[:, 0]]
            is_safe = bool((values <= threshold).all())
            if not is_safe:
                self.get_logger().warning(
                    f'check_footprint_sweep (edges only): {len(all_points)} pixels 中有障碍'
                )
            return is_safe

        # 9. 策略3：正常下采样
        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y

        long_current = long_min
        while long_current <= long_max:
            short_current = short_min
            while short_current <= short_max:
                wx = robot_x + long_current * long_axis_x + short_current * short_axis_x
                wy = robot_y + long_current * long_axis_y + short_current * short_axis_y

                px = int((wx - origin_x) / resolution)
                py = int((wy - origin_y) / resolution)
                px = max(0, min(width - 1, px))
                py = max(0, min(height - 1, py))

                if costmap[py, px] > threshold:
                    self.get_logger().info(
                        f'check_footprint_sweep (sampled): obstacle at ({wx:.2f}, {wy:.2f})')
                    return False

                short_current += short_edge_step
            long_current += long_edge_step

        return True

    # 判断两个点之间的角度差
    def angle_diff(self, a, b, use_abs=True):
        """
        判断以指定像素点为中心、半径为 x 像素的圆形区域内所有像素值是否都大于 0
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
        """
        # 计算相加后的角度
        new_angle = current_angle + rotation_angle
        # 将结果限制在 [-pi, pi] 范围内
        while new_angle > math.pi:
            new_angle -= 2 * math.pi
        while new_angle < -math.pi:
            new_angle += 2 * math.pi
        return new_angle

    
    def check_path_is_free(self, costmap, robot_pixel, target_pixel, threshold=253):
        """
        检查机器人到候选点的连线上是否无障碍
        """
        bresenham_points = self.bresenham(
            robot_pixel[0], robot_pixel[1],
            target_pixel[0], target_pixel[1],
            costmap
        )
        values = np.array([costmap[p[1], p[0]] for p in bresenham_points])
        return bool((values <= threshold).all())

    def get_polygon_edges(self, vertices):
        """返回四边形的四条边 [(p0,p1), (p1,p2), (p2,p3), (p3,p0)]"""
        return [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[3]),
            (vertices[3], vertices[0]),
        ]

    def get_short_edges(self, vertices):
        """返回两条短边"""
        edges = self.get_polygon_edges(vertices)
        edge_infos = []
        for edge in edges:
            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            length = np.linalg.norm(p2 - p1)
            edge_infos.append((edge, length))

        edge_infos.sort(key=lambda x: x[1])  # 从短到长
        return [edge_infos[0][0], edge_infos[1][0]]

    def get_forward_short_edge(self, robot_x, robot_y, vertices, k):
        """按搜索方向 k 找前方短边"""
        short_edges = self.get_short_edges(vertices)
        dir_vec = np.array([math.cos(k), math.sin(k)])
        robot = np.array([robot_x, robot_y])

        best_edge = None
        best_proj = -float('inf')

        for edge in short_edges:
            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            mid = (p1 + p2) / 2.0
            proj = np.dot(mid - robot, dir_vec)
            if proj > best_proj:
                best_proj = proj
                best_edge = edge

        return best_edge

    def is_point_beyond_edge(self, point, edge, polygon_center):
        """point 是否越过 edge 到多边形外侧"""
        p = np.array(point, dtype=float)
        a = np.array(edge[0], dtype=float)
        b = np.array(edge[1], dtype=float)
        c = np.array(polygon_center, dtype=float)

        edge_vec = b - a
        normal1 = np.array([-edge_vec[1], edge_vec[0]], dtype=float)
        normal2 = -normal1

        if np.dot(c - a, normal1) >= 0:
            inward_normal = normal1
        else:
            inward_normal = normal2

        return np.dot(p - a, inward_normal) < 0

    def bresenham(self, current_x, current_y, target_x, target_y, map_array):
        """
        提取两点连线上的所有像素点

        """
        pixels = []
        dx = abs(target_x - current_x)
        dy = abs(target_y - current_y)
        sx = 1 if current_x < target_x else -1
        sy = 1 if current_y < target_y else -1
        err = dx - dy

        x, y = current_x, current_y
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
    