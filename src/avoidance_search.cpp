#include "find_free_space/car_avoidance_point_server.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace find_free_space
{

namespace
{

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
  tf2::Quaternion quaternion;
  quaternion.setRPY(0.0, 0.0, yaw);
  return tf2::toMsg(quaternion);
}

// 沿通道长边方向，选择与来车方向背对的 yaw，结果不做归一化
double computeAvoidanceYaw(double k, const Vec2 & robot, const Vec2 & car)
{
  const Vec2 to_car = car - robot;
  const double dot = std::cos(k) * to_car.x() + std::sin(k) * to_car.y();
  return dot < 0.0 ? k : k + M_PI;
}

}  // namespace

std::optional<CarAvoidancePointServer::SearchContext> CarAvoidancePointServer::prepareAvoidanceSearch(
  const Task & task)
{
  SearchContext ctx;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    ctx.costmap = global_costmap_;
  }
  if (!ctx.costmap) {
    RCLCPP_ERROR(get_logger(), "全局代价地图未收到");
    return std::nullopt;
  }
  const auto & map_info = ctx.costmap->metadata;
  ctx.map.data = ctx.costmap->data.data();
  ctx.map.origin_x = map_info.origin.position.x;
  ctx.map.origin_y = map_info.origin.position.y;
  ctx.map.resolution = map_info.resolution;
  ctx.map.width = static_cast<int>(map_info.size_x);
  ctx.map.height = static_cast<int>(map_info.size_y);
  if (ctx.costmap->data.size() != static_cast<size_t>(ctx.map.width) * static_cast<size_t>(ctx.map.height)) {
    throw std::runtime_error("代价地图数据长度与 size_x * size_y 不一致");
  }
  RCLCPP_INFO_ONCE(get_logger(), "origin_x: %.3f", ctx.map.origin_x);
  RCLCPP_INFO_ONCE(get_logger(), "origin_y: %.3f", ctx.map.origin_y);
  RCLCPP_INFO_ONCE(get_logger(), "resolution: %.3f", ctx.map.resolution);
  RCLCPP_INFO_ONCE(get_logger(), "width: %d", ctx.map.width);
  RCLCPP_INFO_ONCE(get_logger(), "height: %d", ctx.map.height);

  ctx.robot = Vec2(task.robot_pose.pose.position.x, task.robot_pose.pose.position.y);
  ctx.robot_yaw = tf2::getYaw(task.robot_pose.pose.orientation);
  ctx.car = Vec2(task.goal->car_pose.pose.position.x, task.goal->car_pose.pose.position.y);

  RCLCPP_INFO(get_logger(), "寻找最近的边界...");
  const auto nearest_boundary = findNearestBoundary(ctx.robot, task.vertices);
  if (!nearest_boundary) {
    throw std::runtime_error("通道两条最长边长度都为 0，求不出最近长边");
  }
  ctx.nearest_boundary = *nearest_boundary;
  const Vec2 & nb0 = ctx.nearest_boundary.first;
  const Vec2 & nb1 = ctx.nearest_boundary.second;
  RCLCPP_INFO(get_logger(), "nearest_boundary: [(%.3f, %.3f), (%.3f, %.3f)]", nb0.x(), nb0.y(), nb1.x(), nb1.y());

  // 计算搜索方向 k
  ctx.k = std::atan2(nb0.y() - nb1.y(), nb0.x() - nb1.x());
  RCLCPP_INFO(get_logger(), "k_radian: %.3f, k_degree: %.3f", ctx.k, ctx.k / M_PI * 180.0);
  RCLCPP_INFO(get_logger(), "robot: (%.3f, %.3f)", ctx.robot.x(), ctx.robot.y());
  RCLCPP_INFO(get_logger(), "car: (%.3f, %.3f)", ctx.car.x(), ctx.car.y());

  publishDebugMarkers(ctx.robot, ctx.car, ctx.nearest_boundary);

  // 根据车的位置调整 k 方向
  const double car_robot_k = std::atan2(ctx.robot.y() - ctx.car.y(), ctx.robot.x() - ctx.car.x());
  const double k_diff = angleDiff(ctx.k, car_robot_k);
  RCLCPP_INFO(get_logger(), "k: %.3f, car_robot_k: %.3f, k_diff: %.3f", ctx.k, car_robot_k, k_diff);
  if (k_diff > M_PI / 2.0) {
    ctx.k = addAngles(ctx.k, M_PI);
  }

  ctx.k_diff2 = angleDiff(ctx.k, ctx.robot_yaw);
  RCLCPP_INFO(get_logger(), "k: %.3f, k_robot: %.3f, k_diff2: %.3f", ctx.k, ctx.robot_yaw, ctx.k_diff2);
  return ctx;
}

std::vector<CarAvoidancePointServer::PoseStamped> CarAvoidancePointServer::filterExternalStopPoints(
  const SearchContext & ctx)
{
  geometry_msgs::msg::PoseArray::ConstSharedPtr stop_points;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stop_points = vehicle_avoidance_stop_points_;
  }
  if (!stop_points || stop_points->poses.empty()) {
    return {};
  }

  const double cos_yaw = std::cos(ctx.robot_yaw);
  const double sin_yaw = std::sin(ctx.robot_yaw);
  auto local_x = [&](const Vec2 & world) {
      const Vec2 delta = world - ctx.robot;
      return delta.x() * cos_yaw + delta.y() * sin_yaw;
    };
  const double car_local_x = local_x(ctx.car);

  std::vector<std::pair<double, PoseStamped>> results;
  for (const auto & pose : stop_points->poses) {
    const Vec2 point(pose.position.x, pose.position.y);
    const double dist = (point - ctx.robot).norm();
    if (dist > external_point_max_distance_) {
      continue;
    }
    // base_link 下与车在 x 轴异侧才保留
    if (car_local_x * local_x(point) >= 0.0) {
      RCLCPP_INFO(get_logger(), "外部停车点 (%.2f, %.2f) 与车同侧，过滤掉", point.x(), point.y());
      continue;
    }
    PoseStamped pose_stamped;
    pose_stamped.header.stamp = now();
    pose_stamped.header.frame_id = "map";
    pose_stamped.pose.position.x = point.x();
    pose_stamped.pose.position.y = point.y();
    pose_stamped.pose.position.z = 0.0;
    // 先保留外部点的原始方向，通道内/外的方向处理放到 findAvoidancePoint 中决定
    pose_stamped.pose.orientation = pose.orientation;
    results.emplace_back(dist, pose_stamped);
  }
  std::stable_sort(results.begin(), results.end(), [](const auto & lhs, const auto & rhs) {
      return lhs.first < rhs.first;
    });
  RCLCPP_INFO(get_logger(), "外部停车点过滤后剩余: %zu个", results.size());

  std::vector<PoseStamped> poses;
  for (auto & result : results) {
    poses.push_back(std::move(result.second));
  }
  return poses;
}

std::vector<Vec2> CarAvoidancePointServer::searchCellCenters(const MapView & map, const Ring & find_vertices) const
{
  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  for (const auto & vertex : find_vertices) {
    min_x = std::min(min_x, vertex.x());
    max_x = std::max(max_x, vertex.x());
    min_y = std::min(min_y, vertex.y());
    max_y = std::max(max_y, vertex.y());
  }
  auto to_cell = [](double value, double origin, double resolution, int size) {
      return std::clamp(static_cast<int>(std::floor((value - origin) / resolution)), 0, size - 1);
    };
  const int x0 = to_cell(min_x, map.origin_x, map.resolution, map.width);
  const int x1 = to_cell(max_x, map.origin_x, map.resolution, map.width);
  const int y0 = to_cell(min_y, map.origin_y, map.resolution, map.height);
  const int y1 = to_cell(max_y, map.origin_y, map.resolution, map.height);

  std::vector<Vec2> centers;
  for (int cell_y = y0; cell_y <= y1; ++cell_y) {
    for (int cell_x = x0; cell_x <= x1; ++cell_x) {
      const Vec2 center(
        map.origin_x + (cell_x + 0.5) * map.resolution,
        map.origin_y + (cell_y + 0.5) * map.resolution);
      if (pointInConvex(center, find_vertices)) {
        centers.push_back(center);
      }
    }
  }
  return centers;
}

std::optional<CarAvoidancePointServer::PoseStamped> CarAvoidancePointServer::findAvoidancePoint(const Task & task)
{
  const auto prepared = prepareAvoidanceSearch(task);
  if (!prepared) {
    return std::nullopt;
  }
  const SearchContext & ctx = *prepared;
  const Vec2 & robot = ctx.robot;
  const Edge & nearest_boundary = ctx.nearest_boundary;
  const double k = ctx.k;

  // ===== 优先尝试外部停车点 =====
  const auto external_candidates = filterExternalStopPoints(ctx);
  if (!external_candidates.empty()) {
    RCLCPP_INFO(get_logger(), "尝试外部停车点，共%zu个", external_candidates.size());
    std::vector<PoseStamped> valid_external;
    for (auto ext_pose : external_candidates) {
      const Vec2 point(ext_pose.pose.position.x, ext_pose.pose.position.y);
      const bool ext_is_inside = pointInPolygon(point, task.vertices);
      if (ext_is_inside) {
        // 通道内：先按避让方向覆盖 yaw，再校验
        ext_pose.pose.orientation = yawToQuaternion(computeAvoidanceYaw(k, robot, ctx.car));
      } else {
        RCLCPP_INFO(get_logger(), "外部停车点(%.2f, %.2f) 在通道外，将跳过服务检查", point.x(), point.y());
      }
      std::string reason;
      if (validateCandidate(task, ctx, ext_pose, !ext_is_inside, nullptr, reason)) {
        valid_external.push_back(ext_pose);
      } else {
        RCLCPP_INFO(get_logger(), "外部停车点(%.2f, %.2f) %s", point.x(), point.y(), reason.c_str());
      }
    }

    if (!valid_external.empty()) {
      // 选离最近长边最近的点
      auto best = *std::min_element(valid_external.begin(), valid_external.end(),
          [&](const PoseStamped & lhs, const PoseStamped & rhs) {
            return distancePointToSegment(
              Vec2(lhs.pose.position.x, lhs.pose.position.y), nearest_boundary.first, nearest_boundary.second) <
            distancePointToSegment(
              Vec2(rhs.pose.position.x, rhs.pose.position.y), nearest_boundary.first, nearest_boundary.second);
          });
      const Vec2 best_point(best.pose.position.x, best.pose.position.y);
      if (pointInPolygon(best_point, task.vertices)) {
        best.pose.position.z = 20.0;
        RCLCPP_INFO(get_logger(),
          "外部停车点通过校验共%zu个，选择最贴近长边: (%.2f, %.2f)，该点在通道内，使用避让方向，z = 20.0",
          valid_external.size(), best_point.x(), best_point.y());
      } else {
        // 通道外的固定停靠点：方向值保持不变
        best.pose.position.z = 30.0;
        RCLCPP_INFO(get_logger(),
          "外部停车点通过校验共%zu个，选择最贴近长边: (%.2f, %.2f)，该点在通道外，方向保持不变，z = 30.0",
          valid_external.size(), best_point.x(), best_point.y());
        RCLCPP_INFO(get_logger(), "通道外外部停靠点已选用，map yaw=%.1f°",
          radToDeg(tf2::getYaw(best.pose.orientation)));
      }
      return best;
    }
    RCLCPP_INFO(get_logger(), "有外部停车点，但是所有外部停车点在筛选判断以后均不满足，回退自搜索");
  } else {
    RCLCPP_INFO(get_logger(), "话题无有效外部停车点，使用自搜索");
  }

  // ===== 自搜索：沿 k 方向构造搜索框 =====
  const Vec2 k_dir(std::cos(k), std::sin(k));
  Vec2 robot_1;
  Vec2 robot_2;
  if (ctx.k_diff2 < M_PI / 2.0) {
    RCLCPP_INFO(get_logger(), "在机器人前方避车, 额外增加%f米搜索距离", search_radius_extra_dis_);
    robot_1 = robot + k_dir * (search_radius_max_ + search_radius_extra_dis_);
    robot_2 = robot + k_dir * (search_radius_min_ + search_radius_extra_dis_);
  } else {
    RCLCPP_INFO(get_logger(), "在机器人后方避车");
    robot_1 = robot + k_dir * search_radius_max_;
    robot_2 = robot + k_dir * search_radius_min_;
  }

  const double passage_width = minEdgeLength(task.vertices);
  RCLCPP_INFO(get_logger(), "通道宽度: %.3f, 半宽: %.3f", passage_width, passage_width / 2.0);

  const double offset_min = outside_min_;
  const double offset_max = outside_max_;
  RCLCPP_INFO(get_logger(), "当前搜索偏移: offset_min=%.2f, offset_max=%.2f", offset_min, offset_max);

  std::optional<Vec2> p1_near;
  std::optional<Vec2> p2_near;
  std::optional<Vec2> p1_far;
  std::optional<Vec2> p2_far;
  const Vec2 & nb0 = nearest_boundary.first;
  const Vec2 & nb1 = nearest_boundary.second;
  if (pointInPolygon(robot, task.vertices)) {
    p1_near = findIntersection(nb0, nb1, robot_1, offset_min);
    p2_near = findIntersection(nb0, nb1, robot_2, offset_min);
    p1_far = findIntersection(nb0, nb1, robot_1, offset_max);
    p2_far = findIntersection(nb0, nb1, robot_2, offset_max);
  } else {
    // 机器人在通道外：找最近边，并记录它的长度
    double min_dis = std::numeric_limits<double>::infinity();
    double nearest_edge_len = 0.0;
    for (size_t i = 0; i < 4; ++i) {
      const Vec2 & p1 = task.vertices[i];
      const Vec2 & p2 = task.vertices[(i + 1) % 4];
      const double d = distancePointToSegment(robot, p1, p2);
      if (d < min_dis) {
        min_dis = d;
        nearest_edge_len = (p2 - p1).norm();
      }
    }
    const double short_edge_len = minEdgeLength(task.vertices);
    // 最近边长度接近短边长度 → 从短边出去；明显更长 → 从长边出去
    const bool nearest_is_long = nearest_edge_len > short_edge_len * 1.5;
    const double dis_robot_to_nearest_bound = min_dis;

    double offset_min_local = offset_min;
    if (offset_min < dis_robot_to_nearest_bound && dis_robot_to_nearest_bound < offset_max) {
      offset_min_local = dis_robot_to_nearest_bound;
    } else if (nearest_is_long && dis_robot_to_nearest_bound >= stop_in_place_min_dist_) {
      // 从长边出去、且离边够远 → 路肩位置，可原地停
      RCLCPP_INFO(get_logger(), "机器人在长边外侧 %.2fm，原地停车", dis_robot_to_nearest_bound);
      PoseStamped ret_pose;
      ret_pose.header.stamp = now();
      ret_pose.header.frame_id = "map";
      ret_pose.pose.position.x = robot.x();
      ret_pose.pose.position.y = robot.y();
      const double target_angle = radToDeg(std::atan2(nb1.y() - nb0.y(), nb1.x() - nb0.x()));
      const double ret_pose_yaw = degToRad(
        adjustAngle(Vec2(std::cos(ctx.robot_yaw), std::sin(ctx.robot_yaw)), target_angle));
      ret_pose.pose.orientation = yawToQuaternion(ret_pose_yaw);
      ret_pose.pose.position.z = 10.0;
      return ret_pose;
    } else {
      RCLCPP_INFO(get_logger(), "出口边是%s, 距离%.2fm，不满足原地停条件，继续搜索",
        nearest_is_long ? "长边" : "短边", dis_robot_to_nearest_bound);
    }

    p1_near = findIntersection(nb0, nb1, robot_1, -offset_min_local);
    p2_near = findIntersection(nb0, nb1, robot_2, -offset_min_local);
    p1_far = findIntersection(nb0, nb1, robot_1, -offset_max);
    p2_far = findIntersection(nb0, nb1, robot_2, -offset_max);
  }

  if (!p1_near || !p2_near || !p1_far || !p2_far) {
    RCLCPP_INFO(get_logger(), "findIntersection 返回 None，放弃搜索");
    return std::nullopt;
  }

  // 搜索框最前面的短边不能越过通道前方短边
  const auto forward_short_edge = forwardShortEdge(robot, task.vertices, k);
  if (!forward_short_edge) {
    RCLCPP_ERROR(get_logger(), "求不出通道前方短边");
    return std::nullopt;
  }
  Vec2 polygon_center = Vec2::Zero();
  for (const auto & vertex : task.vertices) {
    polygon_center += vertex;
  }
  polygon_center /= static_cast<double>(task.vertices.size());
  if (isPointBeyondEdge(*p1_near, *forward_short_edge, polygon_center) ||
    isPointBeyondEdge(*p1_far, *forward_short_edge, polygon_center))
  {
    RCLCPP_ERROR(get_logger(), "搜索框构造失败：搜索框最前面的短边越过了通道前方短边");
    return std::nullopt;
  }

  const Ring find_vertices = sortQuadrilateralVertices({*p1_near, *p1_far, *p2_far, *p2_near});
  RCLCPP_INFO(get_logger(), "find_vertices: [(%.3f, %.3f),(%.3f, %.3f),(%.3f, %.3f),(%.3f, %.3f)]",
    find_vertices[0].x(), find_vertices[0].y(), find_vertices[1].x(), find_vertices[1].y(),
    find_vertices[2].x(), find_vertices[2].y(), find_vertices[3].x(), find_vertices[3].y());
  publishDebugMarkers(robot, ctx.car, nearest_boundary, &find_vertices);

  // 搜索框内每个栅格中心都是候选点，按离机器人从近到远检查
  auto centers = searchCellCenters(ctx.map, find_vertices);
  std::stable_sort(centers.begin(), centers.end(), [&](const Vec2 & lhs, const Vec2 & rhs) {
      return (lhs - robot).squaredNorm() < (rhs - robot).squaredNorm();
    });
  RCLCPP_INFO(get_logger(), "搜索框内一共%zu个候选栅格", centers.size());

  // 搜索框比栅格还窄时框内可能没有栅格中心，至少保留搜索框中心这一个候选点
  const bool use_box_center = centers.empty();
  if (use_box_center) {
    Vec2 box_center = Vec2::Zero();
    for (const auto & vertex : find_vertices) {
      box_center += vertex;
    }
    centers.push_back(box_center / static_cast<double>(find_vertices.size()));
    RCLCPP_INFO(get_logger(), "搜索框内没有栅格中心，使用搜索框中心(%.3f, %.3f)作为候选点",
      centers.back().x(), centers.back().y());
  }

  const double final_yaw = computeAvoidanceYaw(k, robot, ctx.car);
  const auto orientation = yawToQuaternion(final_yaw);
  const auto kernel = buildFootprintKernel(ctx.map, footprint(), final_yaw);
  const FootprintKernel * kernel_ptr = use_box_center ? nullptr : &kernel;
  std::map<std::string, int> rejected;
  auto log_rejected = [&]() {
      for (const auto & item : rejected) {
        RCLCPP_INFO(get_logger(), "候选栅格不通过：%s %d个", item.first.c_str(), item.second);
      }
    };
  for (const auto & center : centers) {
    PoseStamped candidate;
    candidate.header.stamp = now();
    candidate.header.frame_id = "map";
    candidate.pose.position.x = center.x();
    candidate.pose.position.y = center.y();
    candidate.pose.orientation = orientation;
    std::string reason;
    if (validateCandidate(task, ctx, candidate, false, kernel_ptr, reason)) {
      log_rejected();
      candidate.pose.position.z = 10.0;
      return candidate;
    }
    ++rejected[reason];
  }

  log_rejected();
  RCLCPP_INFO(get_logger(), "当前搜索区域没有找到避让点");
  return std::nullopt;
}

}  // namespace find_free_space
