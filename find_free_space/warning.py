#!/usr/bin/env python3
"""
避车停靠点搜索 - 外部环境模拟 / 功能测试脚本
================================================================

这个脚本用来给 find_free_space_action_server 提供完整外部输入，
并按【测试模式】依次验证下面这些功能：

  1. 机器人 base_link 不在通道内 -> 直接找不到 (Action aborted)
  2. 机器人不在通道内但发了外部停车点 -> 仍 aborted (验证外部点不影响早期 abort)
  3. 通道外的外部固定点 -> 不请求 /check_car_passable 服务
  4. 通道内的外部固定点 -> 仍然请求服务
  5. 自搜索找到的候选点 -> 仍然请求服务
  6. 搜索框最前面的短边越过通道前方短边 -> 搜索框构造失败 -> aborted

  本脚本通过参数 test_mode 切换测试场景，一次只跑一个场景。
  # 场景1: 机器人不在通道内, 应该直接 aborted
  python3 warning.py --ros-args -p test_mode:=robot_outside_abort

  # 场景2: 机器人不在通道内但发了外部停车点, 仍应 aborted
  python3 warning.py --ros-args -p test_mode:=robot_outside_with_external

  # 场景3: 通道外外部固定点, 不请求服务 (服务调用次数应为 0)
  python3 warning.py --ros-args -p test_mode:=external_outside_skip_service

  # 场景4: 通道内外部固定点, 仍然请求服务 (服务调用次数 > 0)
  python3 warning.py --ros-args -p test_mode:=external_inside_need_service

  # 场景5: 自搜索点仍然请求服务 (服务调用次数 > 0)
  python3 warning.py --ros-args -p test_mode:=self_search_need_service

  # 场景6: 搜索框前边越过前方短边, 应该 aborted (服务调用次数应为 0)
  python3 warning.py --ros-args -p test_mode:=search_box_exceed_front_short_edge


================================================================
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


# 所有支持的测试模式 + 预期结果说明 (用于运行时打印, 方便对照)
TEST_MODE_DESC = {
    'robot_outside_abort':
        '机器人不在通道内 -> 预期: Action aborted, 服务调用次数 = 0',
    'robot_outside_with_external':
        '机器人不在通道内但发了外部停车点 -> 预期: 仍 aborted, 服务调用次数 = 0',
    'external_outside_skip_service':
        '通道外外部固定点 -> 预期: 不请求服务(次数=0), Action 成功, z=30',
    'external_inside_need_service':
        '通道内外部固定点 -> 预期: 请求服务(次数>0), Action 成功, z=20',
    'self_search_need_service':
        '自搜索点 -> 预期: 请求服务(次数>0), 成功时 z=10 (依赖地图是否有空闲区)',
    'search_box_exceed_front_short_edge':
        '搜索框前边越过前方短边 -> 预期: Action aborted, 服务调用次数 = 0',
}


class SimulateAvoidanceEnv(Node):
    """
    模拟外部环境, 按 test_mode 驱动 find_free_space_action_server 执行不同测试场景
    """

    def __init__(self):
        super().__init__('simulate_avoidance_env')
        self.get_logger().info('=== 避车环境模拟 / 功能测试节点启动 ===')

        # ========== 参数 ==========
        # 通道几何 / 来车几何
        self.declare_parameter('passage_width', 3.0)          # 通道宽度(短边) m
        self.declare_parameter('passage_length', 12.0)        # 通道长度(长边) m
        self.declare_parameter('car_distance_ahead', 5.0)     # 来车在机器人前方多远 m
        self.declare_parameter('car_width', 2.0)              # 来车宽度 m
        self.declare_parameter('car_length', 4.5)             # 来车长度 m

        # 测试模式
        self.declare_parameter('test_mode', 'external_outside_skip_service')

        # 模拟服务返回策略: 前 N 次服务调用返回 True, 之后 False
        # (用于区分"有没有真的发起服务调用")
        self.declare_parameter('service_return_true_count', 2)

        # 话题 / 服务 / action 名
        self.declare_parameter('action_name', '/find_car_avoidance_point_action')
        self.declare_parameter('stop_points_topic', '/vehicle_avoidance_stop_points')
        self.declare_parameter('check_service_name', '/check_car_passable')

        # robot_outside_abort 模式: 通道整体往前平移多远(让机器人落在通道外) m
        self.declare_parameter('robot_outside_shift', 20.0)
        # search_box_exceed 模式: 把通道长度压到多短(让搜索框前边越界) m
        self.declare_parameter('short_passage_length', 4.0)

        self.passage_width = self.get_parameter('passage_width').value
        self.passage_length = self.get_parameter('passage_length').value
        self.car_distance_ahead = self.get_parameter('car_distance_ahead').value
        self.car_width = self.get_parameter('car_width').value
        self.car_length = self.get_parameter('car_length').value
        self.test_mode = self.get_parameter('test_mode').value
        self.service_return_true_count = self.get_parameter('service_return_true_count').value
        self.action_name = self.get_parameter('action_name').value
        self.stop_points_topic = self.get_parameter('stop_points_topic').value
        self.check_service_name = self.get_parameter('check_service_name').value
        self.robot_outside_shift = self.get_parameter('robot_outside_shift').value
        self.short_passage_length = self.get_parameter('short_passage_length').value

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

        # ========== 发布外部停车点 ==========
        qos_stop_points = QoSProfile(depth=1)
        qos_stop_points.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos_stop_points.reliability = ReliabilityPolicy.RELIABLE
        self.stop_points_pub = self.create_publisher(
            PoseArray, self.stop_points_topic, qos_stop_points
        )

        # ========== Action Client ==========
        self.action_client = ActionClient(
            self, FindCarAvoidancePoint, self.action_name, callback_group=cb_action
        )

        # ========== 定时器 ==========
        self.pose_timer = self.create_timer(0.1, self._update_pose, callback_group=cb_timer)
        self.main_timer = self.create_timer(3.0, self._main_tick, callback_group=cb_timer)

        # 启动时打印当前测试模式
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'当前测试模式: {self.test_mode}')
        self.get_logger().info(f'模式说明: {TEST_MODE_DESC.get(self.test_mode, "未知模式!")}')
        self.get_logger().info(f'模拟服务策略: 前 {self.service_return_true_count} 次返回 True, 之后 False')
        self.get_logger().info('=' * 60)
        self.get_logger().info('等待机器人 TF 和 Action Server 就绪...')

    # ==================== 模拟服务 ====================

    def _check_car_passable_callback(self, request, response):
        """
        模拟 /check_car_passable 服务。
        关键作用: 通过 service_call_count 统计"到底有没有真的发起服务调用",
        用来验证"通道外点不请求服务""通道内点请求服务"。
        """
        self.service_call_count += 1
        robot_pos = request.robot_pose.pose.position
        car_pos = request.car_pose.pose.position

        passable = self.service_call_count <= self.service_return_true_count
        response.is_car_passable = passable
        self.get_logger().info(
            f'[服务调用 #{self.service_call_count}] '
            f'机器人=({robot_pos.x:.2f}, {robot_pos.y:.2f}), '
            f'车=({car_pos.x:.2f}, {car_pos.y:.2f}) '
            f'=> 返回 {"True(可通过)" if passable else "False(不可通过)"}'
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
        self.get_logger().info(f'条件就绪, 开始执行测试: {self.test_mode}')
        self.get_logger().info('=' * 60)

        robot_x = self.robot_pose.pose.position.x
        robot_y = self.robot_pose.pose.position.y
        robot_yaw = self._get_yaw(self.robot_pose)
        self.get_logger().info(
            f'机器人位置: ({robot_x:.3f}, {robot_y:.3f}), yaw={math.degrees(robot_yaw):.1f}°'
        )

        # 步骤1: 发布外部停车点 (按模式定制)
        self.get_logger().info('--- 步骤1: 发布外部停车点 ---')
        self._publish_stop_points(robot_x, robot_y, robot_yaw)
        time.sleep(1.5)  # 等话题传播

        # 步骤2: 构造并发送 Action Goal (按模式定制通道)
        self.get_logger().info('--- 步骤2: 发送 Action Goal ---')
        self._send_action_goal(robot_x, robot_y, robot_yaw)

    def _publish_stop_points(self, robot_x, robot_y, robot_yaw):
        """
        按测试模式发布不同的外部停车点。

        坐标说明:
          along   = 沿机器人朝向(通道长边方向)的偏移, 负数=机器人后方
          lateral = 侧向(通道短边方向)的偏移
          half_w  = 通道半宽; lateral 超过 half_w 即"通道外"

        注意: server 端会把"与来车同侧"的外部点过滤掉, 来车在机器人前方,
        所以这里有效的外部点都放在机器人【后方】(along 为负)。
        """
        cos_d = math.cos(robot_yaw)
        sin_d = math.sin(robot_yaw)
        cos_n = math.cos(robot_yaw + math.pi / 2)
        sin_n = math.sin(robot_yaw + math.pi / 2)
        half_w = self.passage_width / 2.0

        configs = []  # (along, lateral, 描述)

        if self.test_mode == 'external_outside_skip_service':
            # 点放在通道外(横向超出长边), 验证"通道外点不请求服务"
            configs = [
                (-3.0, half_w + 1.0, '通道外右侧点1'),
                (-4.0, half_w + 1.2, '通道外右侧点2'),
            ]
        elif self.test_mode == 'external_inside_need_service':
            # 点放在通道内, 验证"通道内点仍请求服务"
            configs = [
                (-3.0, half_w - 0.5, '通道内右侧点1'),
                (-4.0, 0.0,          '通道内中央点2'),
            ]
        elif self.test_mode == 'self_search_need_service':
            # 故意给"与车同侧"的无效点 -> 会被过滤 -> 逼 server 走自搜索
            configs = [
                (2.0, 0.0, '与车同侧无效点1'),
                (3.0, 0.5, '与车同侧无效点2'),
            ]
        elif self.test_mode == 'robot_outside_with_external':
            # 机器人在通道外, 但仍发外部停车点 -> server 应先判定机器人不在通道内而 abort
            configs = [
                (-3.0, 0.0, '外部点1'),
                (-4.0, 0.5, '外部点2'),
            ]
        elif self.test_mode in ('robot_outside_abort', 'search_box_exceed_front_short_edge'):
            # 这两个模式不需要外部点, 发空数组
            configs = []
        else:
            configs = [(-3.0, half_w - 0.3, '默认点1')]

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for i, (along, lateral, desc) in enumerate(configs):
            px = robot_x + cos_d * along + cos_n * lateral
            py = robot_y + sin_d * along + sin_n * lateral
            pose = Pose()
            pose.position.x = px
            pose.position.y = py
            pose.position.z = 0.0
            # 外部停车点固定给一个显眼的右上45°方向，
            # 这样 rviz 里一眼能看出 server 是用了外部点自带 yaw 还是重算的方向。
            fixed_ext_yaw = math.pi / 4.0
            quat = quaternion_from_euler(0, 0, fixed_ext_yaw)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            msg.poses.append(pose)
            self.get_logger().info(f'  停车点{i+1}: ({px:.3f}, {py:.3f}) - {desc}')

        self.stop_points_pub.publish(msg)
        self.get_logger().info(f'已发布 {len(msg.poses)} 个外部停车点')

    def _send_action_goal(self, robot_x, robot_y, robot_yaw):
        """构造并发送 Action Goal, 通道几何按测试模式定制"""
        cos_d = math.cos(robot_yaw)
        sin_d = math.sin(robot_yaw)
        cos_n = math.cos(robot_yaw + math.pi / 2)
        sin_n = math.sin(robot_yaw + math.pi / 2)

        # --- 来车位姿 (机器人前方, 朝向机器人) ---
        car_pose = PoseStamped()
        car_pose.header.stamp = self.get_clock().now().to_msg()
        car_pose.header.frame_id = 'map'
        car_pose.pose.position.x = robot_x + cos_d * self.car_distance_ahead
        car_pose.pose.position.y = robot_y + sin_d * self.car_distance_ahead
        car_pose.pose.position.z = 0.0
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

        # --- 来车尺寸 ---
        car_size = Vector3()
        car_size.x = self.car_length
        car_size.y = self.car_width
        car_size.z = 1.5

        # --- 通道几何 (按模式定制) ---
        half_len = self.passage_length / 2.0
        half_wid = self.passage_width / 2.0
        center_x = robot_x   # 默认: 通道以机器人为中心 -> 机器人在通道内
        center_y = robot_y

        if self.test_mode in ('robot_outside_abort', 'robot_outside_with_external'):
            # 通道整体往前平移, 让机器人落在通道外 -> server 应判定找不到通道
            center_x = robot_x + cos_d * self.robot_outside_shift
            center_y = robot_y + sin_d * self.robot_outside_shift
            self.get_logger().info(
                f'[{self.test_mode}] 通道中心前移 {self.robot_outside_shift}m, 机器人应在通道外'
            )
        elif self.test_mode == 'search_box_exceed_front_short_edge':
            # 通道压短, 让自搜索框前边越过前方短边
            half_len = self.short_passage_length / 2.0
            self.get_logger().info(
                f'[search_box_exceed] 通道长度压到 {self.short_passage_length}m, 搜索框前边应越界'
            )

        # 四个顶点 (世界坐标)
        vertices_world = [
            (center_x + cos_d * half_len + cos_n * half_wid,
             center_y + sin_d * half_len + sin_n * half_wid),
            (center_x + cos_d * half_len - cos_n * half_wid,
             center_y + sin_d * half_len - sin_n * half_wid),
            (center_x - cos_d * half_len - cos_n * half_wid,
             center_y - sin_d * half_len - sin_n * half_wid),
            (center_x - cos_d * half_len + cos_n * half_wid,
             center_y - sin_d * half_len + sin_n * half_wid),
        ]

        polygon = Polygon()
        for vx, vy in vertices_world:
            pt = Point32()
            pt.x = float(vx)
            pt.y = float(vy)
            pt.z = 0.0
            polygon.points.append(pt)

        self.get_logger().info('通道多边形顶点:')
        for i, (vx, vy) in enumerate(vertices_world):
            self.get_logger().info(f'  V{i}: ({vx:.3f}, {vy:.3f})')

        # --- Goal ---
        goal_msg = FindCarAvoidancePoint.Goal()
        goal_msg.car_pose = car_pose
        goal_msg.car_size = car_size
        goal_msg.polygons = [polygon]

        self.get_logger().info('发送 Action Goal...')
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)

    # ==================== Action 回调 ====================

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Action Goal 被拒绝!')
            return
        self.get_logger().info('Action Goal 已接受, 等待结果...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result()
        status = result.status  # 4=SUCCEEDED, 5=ABORTED, 6=CANCELED

        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Action 执行完毕')
        self.get_logger().info(f'实际状态码: {status} '
                               f'({"成功" if status == 4 else "aborted" if status == 5 else "其他"})')

        if status == 4:
            pose = result.result.pose
            self.get_logger().info(
                f'实际结果: 找到避让点 ({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}), '
                f'z标记={pose.pose.position.z}, 朝向={math.degrees(self._get_yaw(pose)):.1f}°'
            )
        elif status == 5:
            self.get_logger().info('实际结果: Action aborted (未找到有效避让点)')
        else:
            self.get_logger().info(f'实际结果: 其他状态 ({status})')

        self.get_logger().info(f'实际服务总调用次数: {self.service_call_count}')


    # ==================== 工具 ====================

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