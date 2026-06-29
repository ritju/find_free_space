#!/usr/bin/env python3
"""
避车停靠点搜索 - 外部环境模拟脚本

用途：为 find_free_space_action_server 提供完整的外部输入，驱动它执行真实的避让点搜索流程。

本脚本做以下事情：
1. 启动一个模拟的 /check_car_passable 服务（直接返回 True，因为你没有真实服务）
2. 发布外部停车点到 /vehicle_avoidance_stop_points
3. 等待机器人 TF 就绪后，根据机器人当前位置构造通道多边形和来车位姿
4. 发送 FindCarAvoidancePoint Action Goal 给你的真实节点
5. 等待 Action 结果并打印

前置条件：
- Gazebo + nav2 已启动（TF、全局代价地图就绪）
- find_free_space_action_server 已启动
- 本脚本与 find_free_space_action_server 使用相同的话题和服务名

"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import tf2_ros
import math
import time

from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion, Vector3
from geometry_msgs.msg import Polygon, Point32
from capella_ros_msg.action import FindCarAvoidancePoint
from capella_ros_msg.srv import IsCarPassable
from tf_transformations import quaternion_from_euler, euler_from_quaternion


class SimulateAvoidanceEnv(Node):
    """
    模拟外部环境，驱动 find_free_space_action_server 执行完整搜索流程
    """

    def __init__(self):
        super().__init__('simulate_avoidance_env')
        self.get_logger().info('=== 避车环境模拟节点启动 ===')

        # ========== 参数 ==========
        self.declare_parameter('passage_width', 3.0)          # 通道宽度 m
        self.declare_parameter('passage_length', 12.0)        # 通道长度 m
        self.declare_parameter('car_distance_ahead', 5.0)     # 来车在机器人前方多远 m
        self.declare_parameter('car_width', 2.0)              # 来车宽度 m
        self.declare_parameter('car_length', 4.5)             # 来车长度 m
        self.declare_parameter('num_stop_points', 6)          # 发布多少个外部停车点
        self.declare_parameter('service_return_true_count', 2)  # 前N个服务调用返回True
        self.declare_parameter('action_name', '/find_car_avoidance_point_action')
        self.declare_parameter('stop_points_topic', '/vehicle_avoidance_stop_points')
        self.declare_parameter('check_service_name', '/check_car_passable')

        self.passage_width = self.get_parameter('passage_width').value
        self.passage_length = self.get_parameter('passage_length').value
        self.car_distance_ahead = self.get_parameter('car_distance_ahead').value
        self.car_width = self.get_parameter('car_width').value
        self.car_length = self.get_parameter('car_length').value
        self.num_stop_points = self.get_parameter('num_stop_points').value
        self.service_return_true_count = self.get_parameter('service_return_true_count').value
        self.action_name = self.get_parameter('action_name').value
        self.stop_points_topic = self.get_parameter('stop_points_topic').value
        self.check_service_name = self.get_parameter('check_service_name').value

        # ========== 状态 ==========
        self.robot_pose = None
        self.goal_sent = False
        self.service_call_count = 0

        # ========== TF ==========
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ========== 回调组 ==========
        cb_service = ReentrantCallbackGroup()
        cb_action = MutuallyExclusiveCallbackGroup()
        cb_timer = MutuallyExclusiveCallbackGroup()

        # ========== 模拟 /check_car_passable 服务 ==========
        self.check_service = self.create_service(
            IsCarPassable,
            self.check_service_name,
            self._check_car_passable_callback,
            callback_group=cb_service
        )
        self.get_logger().info(f'模拟服务已创建: {self.check_service_name}')
        self.get_logger().info(f'  前 {self.service_return_true_count} 次调用返回 True，之后返回 False')

        # ========== 发布外部停车点 ==========
        qos_stop_points = QoSProfile(depth=1)
        qos_stop_points.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_stop_points.reliability = ReliabilityPolicy.RELIABLE
        self.stop_points_pub = self.create_publisher(
            PoseArray,
            self.stop_points_topic,
            qos_stop_points
        )

        # ========== Action Client ==========
        self.action_client = ActionClient(
            self,
            FindCarAvoidancePoint,
            self.action_name,
            callback_group=cb_action
        )

        # ========== 定时器：获取位姿并触发主逻辑 ==========
        self.pose_timer = self.create_timer(0.1, self._update_pose, callback_group=cb_timer)
        self.main_timer = self.create_timer(3.0, self._main_tick, callback_group=cb_timer)

        self.get_logger().info('等待机器人 TF 和 Action Server 就绪...')

    # ==================== 模拟服务 ====================

    def _check_car_passable_callback(self, request, response):
        """
        模拟 /check_car_passable 服务
        可以控制前 N 次返回 True，模拟"有2个停靠点满足"的场景
        """
        self.service_call_count += 1
        robot_pos = request.robot_pose.pose.position
        car_pos = request.car_pose.pose.position

        if self.service_call_count <= self.service_return_true_count:
            response.is_car_passable = True
            self.get_logger().info(
                f'[服务调用 #{self.service_call_count}] 机器人=({robot_pos.x:.2f}, {robot_pos.y:.2f}), '
                f'车=({car_pos.x:.2f}, {car_pos.y:.2f}) => 返回 True (可通过)'
            )
        else:
            response.is_car_passable = False
            self.get_logger().info(
                f'[服务调用 #{self.service_call_count}] 机器人=({robot_pos.x:.2f}, {robot_pos.y:.2f}), '
                f'车=({car_pos.x:.2f}, {car_pos.y:.2f}) => 返回 False (不可通过)'
            )

        return response

    # ==================== TF 更新 ====================

    def _update_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = 0.0
            pose.pose.orientation = trans.transform.rotation
            self.robot_pose = pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    # ==================== 主逻辑 ====================

    def _main_tick(self):
        if self.goal_sent:
            return

        if self.robot_pose is None:
            self.get_logger().info('等待机器人位姿...', throttle_duration_sec=5.0)
            return

        if not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info(
                f'等待 Action Server: {self.action_name}...', throttle_duration_sec=5.0
            )
            return

        self.goal_sent = True
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('所有条件就绪，开始发送模拟数据')
        self.get_logger().info('=' * 60)

        robot_x = self.robot_pose.pose.position.x
        robot_y = self.robot_pose.pose.position.y
        robot_yaw = self._get_yaw(self.robot_pose)

        self.get_logger().info(
            f'机器人位置: ({robot_x:.3f}, {robot_y:.3f}), yaw={math.degrees(robot_yaw):.1f}°'
        )

        # 步骤1：发布外部停车点
        self.get_logger().info('')
        self.get_logger().info('--- 步骤1：发布外部停车点 ---')
        self._publish_stop_points(robot_x, robot_y, robot_yaw)

        # 等一下让话题传播
        time.sleep(1.5)

        # 步骤2：构造并发送 Action Goal
        self.get_logger().info('')
        self.get_logger().info('--- 步骤2：发送 Action Goal ---')
        self._send_action_goal(robot_x, robot_y, robot_yaw)

    def _publish_stop_points(self, robot_x, robot_y, robot_yaw):
        """发布外部停车点"""
        cos_d = math.cos(robot_yaw)
        sin_d = math.sin(robot_yaw)
        cos_n = math.cos(robot_yaw + math.pi / 2)
        sin_n = math.sin(robot_yaw + math.pi / 2)

        half_w = self.passage_width / 2.0

        # 生成停车点配置：(沿通道方向偏移, 侧向偏移, 描述)
        configs = [
            (-3.0,  half_w - 0.3, '后方3m，靠右边缘'),
            (-4.0,  half_w - 0.5, '后方4m，靠右边缘'),
            (-3.5,  0.0,          '后方3.5m，通道中央'),
            (-5.0,  half_w - 0.2, '后方5m，靠右边缘'),
            (-2.0, -half_w + 0.3, '后方2m，靠左边缘'),
            (-10.0, half_w - 0.3, '后方10m，超距离'),
        ]

        # 如果需要更多点，补齐
        while len(configs) < self.num_stop_points:
            extra_dist = -3.0 - len(configs) * 0.5
            configs.append((extra_dist, half_w - 0.4, f'额外点{len(configs)+1}'))

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for i, (along, lateral, desc) in enumerate(configs[:self.num_stop_points]):
            px = robot_x + cos_d * along + cos_n * lateral
            py = robot_y + sin_d * along + sin_n * lateral

            pose = Pose()
            pose.position.x = px
            pose.position.y = py
            pose.position.z = 0.0
            quat = quaternion_from_euler(0, 0, robot_yaw)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            msg.poses.append(pose)

            self.get_logger().info(f'  停车点{i+1}: ({px:.3f}, {py:.3f}) - {desc}')

        self.stop_points_pub.publish(msg)
        self.get_logger().info(f'已发布 {len(msg.poses)} 个外部停车点')

    def _send_action_goal(self, robot_x, robot_y, robot_yaw):
        """构造并发送 Action Goal"""

        cos_d = math.cos(robot_yaw)
        sin_d = math.sin(robot_yaw)
        cos_n = math.cos(robot_yaw + math.pi / 2)
        sin_n = math.sin(robot_yaw + math.pi / 2)

        # --- 构造来车位姿 ---
        car_pose = PoseStamped()
        car_pose.header.stamp = self.get_clock().now().to_msg()
        car_pose.header.frame_id = 'map'
        car_pose.pose.position.x = robot_x + cos_d * self.car_distance_ahead
        car_pose.pose.position.y = robot_y + sin_d * self.car_distance_ahead
        car_pose.pose.position.z = 0.0
        # 车朝向机器人方向
        car_yaw = robot_yaw + math.pi
        quat = quaternion_from_euler(0, 0, car_yaw)
        car_pose.pose.orientation.x = quat[0]
        car_pose.pose.orientation.y = quat[1]
        car_pose.pose.orientation.z = quat[2]
        car_pose.pose.orientation.w = quat[3]

        self.get_logger().info(
            f'来车位姿: ({car_pose.pose.position.x:.3f}, {car_pose.pose.position.y:.3f}), '
            f'yaw={math.degrees(car_yaw):.1f}°'
        )

        # --- 构造来车尺寸 ---
        car_size = Vector3()
        car_size.x = self.car_length
        car_size.y = self.car_width
        car_size.z = 1.5  # 高度随便给一个

        self.get_logger().info(f'来车尺寸: {self.car_length}m x {self.car_width}m')

        # --- 构造通道多边形 ---
        half_len = self.passage_length / 2.0
        half_wid = self.passage_width / 2.0

        # 四个顶点
        vertices_world = [
            (robot_x + cos_d * half_len + cos_n * half_wid,
             robot_y + sin_d * half_len + sin_n * half_wid),
            (robot_x + cos_d * half_len - cos_n * half_wid,
             robot_y + sin_d * half_len - sin_n * half_wid),
            (robot_x - cos_d * half_len - cos_n * half_wid,
             robot_y - sin_d * half_len - sin_n * half_wid),
            (robot_x - cos_d * half_len + cos_n * half_wid,
             robot_y - sin_d * half_len + sin_n * half_wid),
        ]

        polygon = Polygon()
        for vx, vy in vertices_world:
            pt = Point32()
            pt.x = float(vx)
            pt.y = float(vy)
            pt.z = 0.0
            polygon.points.append(pt)

        self.get_logger().info(f'通道多边形顶点:')
        for i, (vx, vy) in enumerate(vertices_world):
            self.get_logger().info(f'  V{i}: ({vx:.3f}, {vy:.3f})')

        # --- 构造 Goal ---
        goal_msg = FindCarAvoidancePoint.Goal()
        goal_msg.car_pose = car_pose
        goal_msg.car_size = car_size
        goal_msg.polygons = [polygon]

        self.get_logger().info('')
        self.get_logger().info('发送 Action Goal...')
        self.get_logger().info('=' * 60)

        # 发送 Goal
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)

    # ==================== Action 回调 ====================

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Action Goal 被拒绝！')
            return

        self.get_logger().info('Action Goal 已接受，等待结果...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result()
        status = result.status

        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Action 执行完毕')
        self.get_logger().info(f'状态码: {status}')

        # status: 4=SUCCEEDED, 5=ABORTED, 6=CANCELED
        if status == 4:
            pose = result.result.pose
            self.get_logger().info('结果: 成功找到避让点！')
            self.get_logger().info(
                f'  位置: ({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f})'
            )
            self.get_logger().info(f'  z 标记: {pose.pose.position.z}')
            yaw = self._get_yaw(pose)
            self.get_logger().info(f'  朝向: {math.degrees(yaw):.1f}°')
        elif status == 5:
            self.get_logger().info('结果: Action Aborted（未找到有效避让点）')
        else:
            self.get_logger().info(f'结果: 其他状态 ({status})')

        self.get_logger().info(f'服务总调用次数: {self.service_call_count}')
        self.get_logger().info('=' * 60)

    # ==================== 工具函数 ====================

    def _get_yaw(self, pose_stamped):
        o = pose_stamped.pose.orientation
        _, _, yaw = euler_from_quaternion([o.x, o.y, o.z, o.w])
        return yaw


def main(args=None):
    rclpy.init(args=args)
    node = SimulateAvoidanceEnv()
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()