#!/usr/bin/env python3
"""
读取机器人当前位姿，将"车辆"目标设为正前方5m处，
构造合理的通行区域多边形后发送 FindCarAvoidancePoint Action Goal。

同时，复刻 action server 的搜索框计算逻辑，
把搜索框区域发布为禁扫区，验证禁扫区过滤效果。
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros
import math
import time
import numpy as np

from geometry_msgs.msg import PoseStamped, Point32, Quaternion, Point
from geometry_msgs.msg import Polygon as GeoPolygon
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from capella_ros_msg.action import FindCarAvoidancePoint

from garage_utils_msgs.msg import Polygons
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, DurabilityPolicy, QoSDurabilityPolicy, ReliabilityPolicy


class SendGoalAheadNode(Node):
    def __init__(self):
        super().__init__('send_goal_ahead_node')
        self.get_logger().info('=== 启动: 读取机器人坐标 & 发送正前方5m目标 ===')

        # ---------- 可调参数 ----------
        self.goal_distance = 5.0        # 正前方距离(m)
        self.car_length = 4.5           # 车辆长度(m)
        self.car_width = 1.8            # 车辆宽度(m)
        self.car_height = 1.5           # 车辆高度(m)
        self.corridor_half_width = 3.0  # 通道半宽(m)
        self.corridor_length = 15.0     # 通道长度(m)

        # ---------- 搜索框参数（与 action server 一致） ----------
        self.search_radius_min = 3.0
        self.search_radius_max = 4.0
        self.search_radius_extra_dis = 2.0
        self.outside_min = 0.0
        self.outside_max = 0.5

        # ---------- TF2 ----------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------- Action Client ----------
        self.action_client = ActionClient(
            self,
            FindCarAvoidancePoint,
            '/find_car_avoidance_point_action'
        )

        # ---------- 禁扫区发布器 ----------
        qos_transient = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.special_terrain_pub = self.create_publisher(
            Polygons,
            '/cleaning_tool_retraction_areas',
            qos_transient,
        )

        # ---------- 搜索框可视化 Marker ----------
        marker_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
        )
        self.marker_search_rect_pub = self.create_publisher(
            Marker,
            '/marker_test_search_rect',
            marker_qos,
        )

        # 等5秒让TF缓存就绪后发送
        self.timer = self.create_timer(5.0, self.on_timer)

    # =========================================================
    def on_timer(self):
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

        # ---- Step 2: 计算正前方5m的目标点（车辆位置） ----
        target_x = robot_x + self.goal_distance * math.cos(yaw)
        target_y = robot_y + self.goal_distance * math.sin(yaw)

        # ---- Step 3: 构建通道多边形 ----
        corridor_polygon = self.build_corridor_polygon(
            robot_x, robot_y, yaw,
            self.corridor_length,
            self.corridor_half_width
        )

        # 把 corridor_polygon 转成 numpy 数组格式（与 action server 中 vertices 一致）
        vertices = np.array([[pt.x, pt.y] for pt in corridor_polygon.points])

        # ---- Step 4: 对两条长边分别计算搜索框 ----
        search_rects = self.compute_both_search_rectangles(
            robot_x, robot_y, yaw,
            target_x, target_y,  # 车辆位置
            vertices
        )

        if search_rects:
            # ---- Step 5: 把所有搜索框发布为禁扫区 ----
            self.publish_special_terrain(search_rects)

            # ---- Step 5.5: 可视化每个搜索框 ----
            for i, rect in enumerate(search_rects):
                self.publish_search_rect_marker(rect, base_id=i * 10)
        else:
            self.get_logger().warn('无法计算搜索框，跳过禁扫区发布')

        # 等一下让订阅者收到禁扫区
        time.sleep(1.0)

        # ---- Step 6: 构建 Action Goal ----
        goal_msg = FindCarAvoidancePoint.Goal()

        car_pose = PoseStamped()
        car_pose.header.frame_id = 'map'
        car_pose.header.stamp = self.get_clock().now().to_msg()
        car_pose.pose.position.x = target_x
        car_pose.pose.position.y = target_y
        car_pose.pose.position.z = 0.0
        car_pose.pose.orientation = ori
        goal_msg.car_pose = car_pose

        from geometry_msgs.msg import Vector3
        car_size = Vector3()
        car_size.x = self.car_length
        car_size.y = self.car_width
        car_size.z = self.car_height
        goal_msg.car_size = car_size
        goal_msg.polygons = [corridor_polygon]

        self.get_logger().info(
            f'车辆参数: 长={self.car_length}m, 宽={self.car_width}m, '
            f'高={self.car_height}m'
        )

        # ---- Step 7: 发送 Goal ----
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
    #  复刻 action server 搜索框计算逻辑
    # =========================================================
    def compute_both_search_rectangles(self, robot_x, robot_y, robot_yaw,
                                        car_x, car_y, vertices):
        """对两条长边分别计算搜索框，返回两个矩形"""
        edges = [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[3]),
            (vertices[3], vertices[0])
        ]
        edge_lengths = [np.linalg.norm(np.array(e[1]) - np.array(e[0])) for e in edges]
        sorted_edges = sorted(zip(edges, edge_lengths), key=lambda x: -x[1])
        long_edges = [edge for edge, _ in sorted_edges[:2]]

        results = []
        for boundary in long_edges:
            rect = self._compute_rect_for_boundary(
                robot_x, robot_y, robot_yaw, car_x, car_y, boundary
            )
            if rect is not None:
                results.append(rect)
        return results

    def _compute_rect_for_boundary(self, robot_x, robot_y, robot_yaw,
                                    car_x, car_y, nearest_boundary):
        """针对单条边界计算搜索框"""
        p1, p2 = nearest_boundary
        y_ = p1[1] - p2[1]
        x_ = p1[0] - p2[0]
        k = math.atan2(y_, x_)

        y_ = robot_y - car_y
        x_ = robot_x - car_x
        car_robot_k = math.atan2(y_, x_)
        k_diff = self.angle_diff(k, car_robot_k)
        if k_diff > math.pi / 2:
            k = self.add_angles(k, math.pi)

        k_diff2 = self.angle_diff(k, robot_yaw)
        if k_diff2 < math.pi / 2:
            robot_x1 = robot_x + math.cos(k) * (self.search_radius_max + self.search_radius_extra_dis)
            robot_y1 = robot_y + math.sin(k) * (self.search_radius_max + self.search_radius_extra_dis)
            robot_x2 = robot_x + math.cos(k) * (self.search_radius_min + self.search_radius_extra_dis)
            robot_y2 = robot_y + math.sin(k) * (self.search_radius_min + self.search_radius_extra_dis)
        else:
            robot_x1 = robot_x + math.cos(k) * self.search_radius_max
            robot_y1 = robot_y + math.sin(k) * self.search_radius_max
            robot_x2 = robot_x + math.cos(k) * self.search_radius_min
            robot_y2 = robot_y + math.sin(k) * self.search_radius_min

        p1_near = self.findIntersection(p1, p2, [robot_x1, robot_y1], self.outside_min)
        p2_near = self.findIntersection(p1, p2, [robot_x2, robot_y2], self.outside_min)
        p1_far = self.findIntersection(p1, p2, [robot_x1, robot_y1], self.outside_max)
        p2_far = self.findIntersection(p1, p2, [robot_x2, robot_y2], self.outside_max)

        if any(p is None for p in [p1_near, p2_near, p1_far, p2_far]):
            return None

        return self.sort_quadrilateral_vertices([p1_near, p1_far, p2_far, p2_near])

    # =========================================================
    #  辅助方法（从 action server 复刻）
    # =========================================================
    def find_nearest_boundary(self, robot_x, robot_y, vertices):
        """寻找距离机器人最近的长边"""
        edges = [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[3]),
            (vertices[3], vertices[0])
        ]

        edge_lengths = [np.linalg.norm(np.array(e[1]) - np.array(e[0])) for e in edges]
        sorted_edges = sorted(zip(edges, edge_lengths), key=lambda x: -x[1])
        long_edges = [edge for edge, _ in sorted_edges[:2]]

        robot_point = np.array([robot_x, robot_y])
        min_distance = float('inf')
        nearest_boundary = None

        for edge in long_edges:
            p1, p2 = np.array(edge[0]), np.array(edge[1])
            a = p1[1] - p2[1]
            b = p2[0] - p1[0]
            c = p1[0] * p2[1] - p2[0] * p1[1]
            distance = abs(a * robot_point[0] + b * robot_point[1] + c) / math.sqrt(a**2 + b**2)
            if distance < min_distance:
                min_distance = distance
                nearest_boundary = (p1, p2)

        return nearest_boundary

    def findIntersection(self, a, b, c, distance):
        """计算点c关于直线ab偏移distance的点"""
        ab_x = b[0] - a[0]
        ab_y = b[1] - a[1]
        length_ab = math.sqrt(ab_x**2 + ab_y**2)
        if length_ab == 0:
            return None

        unit_ab_x = ab_x / length_ab
        unit_ab_y = ab_y / length_ab
        normal_x = -unit_ab_y
        normal_y = unit_ab_x

        ac_x = c[0] - a[0]
        ac_y = c[1] - a[1]
        projection = ac_x * unit_ab_x + ac_y * unit_ab_y
        foot_x = a[0] + projection * unit_ab_x
        foot_y = a[1] + projection * unit_ab_y

        normal_projection = ac_x * normal_x + ac_y * normal_y
        direction = -1 if normal_projection > 0 else 1

        target_x = foot_x + direction * normal_x * distance
        target_y = foot_y + direction * normal_y * distance
        return (target_x, target_y)

    def is_point_inside_parallelogram(self, robot_x, robot_y, vertices):
        """射线法判断点是否在多边形内部"""
        inside = False
        j = len(vertices) - 1
        for i in range(len(vertices)):
            xi, yi = vertices[i][0], vertices[i][1]
            xj, yj = vertices[j][0], vertices[j][1]
            intersect = ((yi > robot_y) != (yj > robot_y)) and \
                        (robot_x < (xj - xi) * (robot_y - yi) / (yj - yi) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    def dis_point_to_line2(self, x, y, p1_x, p1_y, p2_x, p2_y):
        """点到直线的垂直距离"""
        if p1_x == p2_x:
            return abs(x - p1_x)
        A = p2_y - p1_y
        B = p1_x - p2_x
        C = p2_x * p1_y - p1_x * p2_y
        numerator = abs(A * x + B * y + C)
        denominator = math.sqrt(A**2 + B**2)
        return numerator / denominator

    def angle_diff(self, a, b):
        error = a - b
        if error < -math.pi:
            error += 2 * math.pi
        elif error >= math.pi:
            error -= 2 * math.pi
        return abs(error)

    def add_angles(self, current_angle, rotation_angle):
        new_angle = current_angle + rotation_angle
        while new_angle > math.pi:
            new_angle -= 2 * math.pi
        while new_angle < -math.pi:
            new_angle += 2 * math.pi
        return new_angle

    def sort_quadrilateral_vertices(self, points):
        """对四边形顶点排序"""
        points = np.array(points)
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        top_points = sorted_points[np.argsort(sorted_points[:, 1])[:2]]
        if top_points[0][0] > top_points[1][0]:
            top_points = top_points[::-1]

        bottom_points = sorted_points[np.argsort(sorted_points[:, 1])[2:]]
        if bottom_points[0][0] < bottom_points[1][0]:
            bottom_points = bottom_points[::-1]

        return np.vstack([top_points, bottom_points])

    # =========================================================
    #  发布禁扫区（覆盖搜索框）
    # =========================================================
    def publish_special_terrain(self, search_rects):
        """在搜索框外侧（垂直长边方向）偏移0.5m作为禁扫区，测试footprint重叠检查"""
        msg = Polygons()

        for search_rect in search_rects:
            # 搜索框的长边方向 = 沿着走廊边界的切线方向
            # edge2 是 search_rect[3] - search_rect[0]，即垂直于 search_rect[1] - search_rect[0] 的方向
            edge1 = np.array(search_rect[1]) - np.array(search_rect[0])
            edge2 = np.array(search_rect[3]) - np.array(search_rect[0])
            len1 = np.linalg.norm(edge1)
            len2 = np.linalg.norm(edge2)

            # 找出长边向量
            if len1 > len2:
                long_edge = edge1 / len1
            else:
                long_edge = edge2 / len2

            # 垂直长边方向 = (−long_edge.y, long_edge.x) 指向走廊外侧
            normal = np.array([-long_edge[1], long_edge[0]])

            # 往外侧偏移0.5m，候选点中心不在禁扫区内，但footprint(半长0.96m)能伸入
            offset = normal * 0.5

            polygon = GeoPolygon()
            for x, y in search_rect:
                pt = Point32()
                pt.x = float(x + offset[0])
                pt.y = float(y + offset[1])
                pt.z = 0.0
                polygon.points.append(pt)
            msg.polygons.append(polygon)

        self.special_terrain_pub.publish(msg)
        self.get_logger().info(f'===== 已发布{len(msg.polygons)}个禁扫区（垂直长边偏移0.5m，测试footprint重叠）=====')

    # =========================================================
    #  可视化搜索框（紫色实心 + 黄色线框，双重保险能看见）
    # =========================================================
    def publish_search_rect_marker(self, search_rect, base_id=100):
        now = self.get_clock().now().to_msg()

        # --- 1) 紫色实心填充 ---
        marker_fill = Marker()
        marker_fill.header.frame_id = "map"
        marker_fill.header.stamp = now
        marker_fill.id = base_id
        marker_fill.type = Marker.TRIANGLE_LIST
        marker_fill.action = Marker.ADD
        marker_fill.scale.x = 1.0
        marker_fill.scale.y = 1.0
        marker_fill.scale.z = 1.0
        marker_fill.color.r = 0.58
        marker_fill.color.g = 0.0
        marker_fill.color.b = 0.83
        marker_fill.color.a = 0.5

        pts = search_rect
        for i in range(1, len(pts) - 1):
            for idx in [0, i, i + 1]:
                p = Point()
                p.x = float(pts[idx][0])
                p.y = float(pts[idx][1])
                p.z = 0.0
                marker_fill.points.append(p)

        self.marker_search_rect_pub.publish(marker_fill)

        # --- 2) 黄色线框（醒目） ---
        marker_line = Marker()
        marker_line.header.frame_id = "map"
        marker_line.header.stamp = now
        marker_line.id = base_id + 1
        marker_line.type = Marker.LINE_LIST
        marker_line.action = Marker.ADD
        marker_line.scale.x = 0.15
        marker_line.color.r = 1.0
        marker_line.color.g = 1.0
        marker_line.color.b = 0.0
        marker_line.color.a = 1.0

        n = len(search_rect)
        for i in range(n):
            p_start = Point()
            p_start.x = float(search_rect[i][0])
            p_start.y = float(search_rect[i][1])
            marker_line.points.append(p_start)
            p_end = Point()
            p_end.x = float(search_rect[(i + 1) % n][0])
            p_end.y = float(search_rect[(i + 1) % n][1])
            marker_line.points.append(p_end)

        self.marker_search_rect_pub.publish(marker_line)

    # =========================================================
    #  原有辅助方法
    # =========================================================
    def get_robot_pose(self) -> PoseStamped:
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

    def build_corridor_polygon(self, cx, cy, yaw, length, half_width) -> GeoPolygon:
        front = length * 0.7
        rear = length * 0.3
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        dx_fwd = cos_y
        dy_fwd = sin_y
        dx_right = sin_y
        dy_right = -cos_y

        offsets = [
            (+front, +half_width),
            (-rear,  +half_width),
            (-rear,  -half_width),
            (+front, -half_width),
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
        if status == 4:
            self.get_logger().info(
                f'成功! 避让点: x={pose.pose.position.x:.3f}, y={pose.pose.position.y:.3f}'
            )
        else:
            self.get_logger().warn(f'未成功, status={status}')

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
