#!/usr/bin/env python3
"""
读取机器人当前位姿，将"车辆"目标设为正前方5m处，
构造合理的通行区域多边形后发送 FindCarAvoidancePoint Action Goal。

车辆参数按正常小轿车标准：
  - 长 4.5m, 宽 1.8m, 高 1.5m
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros
import math
import time

from geometry_msgs.msg import PoseStamped, Point32, Quaternion
from geometry_msgs.msg import Polygon as GeoPolygon
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from capella_ros_msg.action import FindCarAvoidancePoint


class SendGoalAheadNode(Node):
    def __init__(self):
        super().__init__('send_goal_ahead_node')
        self.get_logger().info('=== 启动: 读取机器人坐标 & 发送正前方5m目标 ===')

        # ---------- 可调参数 ----------
        self.goal_distance = 5.0        # 正前方距离(m)
        self.car_length = 4.5           # 车辆长度(m)
        self.car_width = 1.8            # 车辆宽度(m)
        self.car_height = 1.5           # 车辆高度(m)
        self.corridor_half_width = 3.0  # 通道半宽(m)，用于构造多边形
        self.corridor_length = 15.0     # 通道长度(m)

        # ---------- TF2 ----------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------- Action Client ----------
        self.action_client = ActionClient(
            self,
            FindCarAvoidancePoint,
            '/find_car_avoidance_point_action'
        )

        # 等2秒让TF缓存就绪后发送
        self.timer = self.create_timer(2.0, self.on_timer)

    # =========================================================
    def on_timer(self):
        """定时器回调：只执行一次"""
        self.timer.cancel()
        self.execute()

    def execute(self):
        # ---- Step 1: 读取机器人当前位姿 ----
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().error('无法获取机器人位姿，退出')
            return

        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y
        ori = robot_pose.pose.orientation
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])

        self.get_logger().info(
            f'机器人当前坐标: x={robot_x:.3f}, y={robot_y:.3f}, '
            f'yaw={math.degrees(yaw):.1f}°'
        )

        # ---- Step 2: 计算正前方5m的目标点 ----
        target_x = robot_x + self.goal_distance * math.cos(yaw)
        target_y = robot_y + self.goal_distance * math.sin(yaw)

        self.get_logger().info(
            f'目标点(正前方{self.goal_distance}m): '
            f'x={target_x:.3f}, y={target_y:.3f}'
        )

        # ---- Step 3: 构建 Action Goal ----
        goal_msg = FindCarAvoidancePoint.Goal()

        # 3a. car_pose — "车"在正前方5m处，朝向与机器人相同
        car_pose = PoseStamped()
        car_pose.header.frame_id = 'map'
        car_pose.header.stamp = self.get_clock().now().to_msg()
        car_pose.pose.position.x = target_x
        car_pose.pose.position.y = target_y
        car_pose.pose.position.z = 0.0
        car_pose.pose.orientation = ori  # 朝向与机器人一致
        goal_msg.car_pose = car_pose

        # 3b. car_size — 正常小轿车尺寸
        from geometry_msgs.msg import Vector3
        car_size = Vector3()
        car_size.x = self.car_length   # 4.5m
        car_size.y = self.car_width    # 1.8m
        car_size.z = self.car_height   # 1.5m
        goal_msg.car_size = car_size

        # 3c. polygons — 沿机器人前方构造一个矩形通道
        corridor_polygon = self.build_corridor_polygon(
            robot_x, robot_y, yaw,
            self.corridor_length,
            self.corridor_half_width
        )
        goal_msg.polygons = [corridor_polygon]

        self.get_logger().info(
            f'车辆参数: 长={self.car_length}m, 宽={self.car_width}m, '
            f'高={self.car_height}m'
        )
        self.get_logger().info(
            f'通道多边形: 长={self.corridor_length}m, '
            f'半宽={self.corridor_half_width}m'
        )

        # ---- Step 4: 发送 Goal ----
        self.get_logger().info('等待 Action Server...')
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action Server 未上线，退出')
            return

        self.get_logger().info('发送 Action Goal...')
        send_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_cb
        )
        send_future.add_done_callback(self.goal_response_cb)

    # =========================================================
    #  辅助方法
    # =========================================================
    def get_robot_pose(self) -> PoseStamped:
        """通过 TF2 获取 map->base_link 变换，返回 PoseStamped"""
        for attempt in range(10):
            try:
                trans = self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time()
                )
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = trans.transform.translation.x
                pose.pose.position.y = trans.transform.translation.y
                pose.pose.position.z = trans.transform.translation.z
                pose.pose.orientation = trans.transform.rotation
                return pose
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'TF查询失败(尝试{attempt+1}/10): {e}')
                time.sleep(0.5)
        return None

    def build_corridor_polygon(
        self, cx, cy, yaw, length, half_width
    ) -> GeoPolygon:
        """
        以机器人位置为中心，沿 yaw 方向构造矩形通道多边形。

        通道坐标示意（机器人车体坐标系）:
            
               p2 ────────────── p1
                |                 |
           ← ← | ← ← robot → → →| → →  (yaw 方向)
                |                 |
               p3 ────────────── p4

        p1: 前方 + 右侧
        p2: 后方 + 右侧  
        p3: 后方 + 左侧
        p4: 前方 + 左侧
        """
        # 前方/后方偏移
        front = length * 0.7   # 机器人前方占70%
        rear = length * 0.3    # 机器人后方占30%

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # 沿 yaw 方向的单位向量
        dx_fwd = cos_y
        dy_fwd = sin_y
        # 垂直于 yaw 的单位向量（右手系：右侧）
        dx_right = sin_y
        dy_right = -cos_y

        points = []
        offsets = [
            (+front, +half_width),   # p1: 前右
            (-rear,  +half_width),   # p2: 后右
            (-rear,  -half_width),   # p3: 后左
            (+front, -half_width),   # p4: 前左
        ]

        polygon = GeoPolygon()
        for fwd_off, right_off in offsets:
            pt = Point32()
            pt.x = float(cx + fwd_off * dx_fwd + right_off * dx_right)
            pt.y = float(cy + fwd_off * dy_fwd + right_off * dy_right)
            pt.z = 0.0
            polygon.points.append(pt)

        self.get_logger().info('通道多边形顶点:')
        for i, pt in enumerate(polygon.points):
            self.get_logger().info(f'  P{i+1}: ({pt.x:.3f}, {pt.y:.3f})')

        return polygon

    # =========================================================
    #  Action 回调
    # =========================================================
    def goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal 被拒绝!')
            return
        self.get_logger().info('Goal 已接受，等待结果...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_cb)

    def result_cb(self, future):
        result = future.result()
        status = result.status
        pose = result.result.pose

        if status == 4:  # SUCCEEDED
            self.get_logger().info(
                f'✅ 成功! 避让点: '
                f'x={pose.pose.position.x:.3f}, '
                f'y={pose.pose.position.y:.3f}'
            )
        else:
            self.get_logger().warn(f'❌ 未成功, status={status}')

    def feedback_cb(self, feedback_msg):
        self.get_logger().info(f'反馈: {feedback_msg.feedback}')


def main(args=None):
    rclpy.init(args=args)
    node = SendGoalAheadNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()