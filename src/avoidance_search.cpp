#include "find_free_space/car_avoidance_point_server.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"

namespace find_free_space
{

namespace
{

// 一个方向有两个朝向。它指向来车时加 180°。
double flipIfFacingCar(double yaw, const Vec2 & robot, const Vec2 & car)
{
  const double to_car_x = car.x() - robot.x();
  const double to_car_y = car.y() - robot.y();
  const double dot = std::cos(yaw) * to_car_x + std::sin(yaw) * to_car_y;
  return dot < 0.0 ? yaw : addAngles(yaw, M_PI);
}

// 通道内部参考点，用来判断搜索框有没有越过短边。
Vec2 shortEdgeInteriorPoint(const Ring & vertices)
{
  Vec2 center = Vec2::Zero();
  for (const auto & point : vertices) {
    center += point;
  }
  if (!vertices.empty()) {
    center /= static_cast<double>(vertices.size());
  }
  return center;
}

// 根据传入的 A 或 B 长边编号，从道路模型中取出对应的长边数据，供后续投影和搜索使用。
const garage_road_model::SideBoundary & avoidanceLongSide(
  const garage_road_model::RoadModelQuery & query, garage_road_model::SideId side)
{
  return side == garage_road_model::SideId::A ? query.model().side_a : query.model().side_b;
}

struct Foot
{
  garage_road_model::SideId side{garage_road_model::SideId::A};
  double s{0.0};
  double distance{0.0};
  double yaw{0.0};
};

// 把点投到指定侧边，返回最近垂足。
std::optional<Foot> nearestFoot(
  const garage_road_model::RoadModelQuery & query,
  garage_road_model::SideId side,
  const Vec2 & point)
{
  if (avoidanceLongSide(query, side).samples.size() < 2) {
    return std::nullopt;
  }
  const auto projection = query.projectOnSide(side, point, 0.0, -1.0);
  const bool has_foot = projection.status == garage_road_model::QueryStatus::Ok ||
    projection.status == garage_road_model::QueryStatus::Ambiguous;
  if (!has_foot || !std::isfinite(projection.distance)) {
    return std::nullopt;
  }
  return Foot{side, projection.s, projection.distance, projection.yaw};
}

}  // namespace

// 用 nearestFoot 把机器人投到两条长边，留下更近的那个垂足。
std::optional<CarAvoidancePointServer::SearchContext> CarAvoidancePointServer::selectAvoidanceFoot(
  const Task & task)
{
  if (!task.road_model) {
    RCLCPP_ERROR(get_logger(), "没有可用曲线，无法找点");
    return std::nullopt;
  }

  SearchContext ctx;
  ctx.robot = Vec2(task.robot_pose.pose.position.x, task.robot_pose.pose.position.y);
  ctx.robot_yaw = tf2::getYaw(task.robot_pose.pose.orientation);
  ctx.car = Vec2(task.goal->car_pose.pose.position.x, task.goal->car_pose.pose.position.y);
  ctx.channel = task.vertices;
  ctx.query = std::make_shared<garage_road_model::RoadModelQuery>(*task.road_model);

  const auto foot_a = nearestFoot(*ctx.query, garage_road_model::SideId::A, ctx.robot);
  const auto foot_b = nearestFoot(*ctx.query, garage_road_model::SideId::B, ctx.robot);
  if (!foot_a && !foot_b) {
    RCLCPP_ERROR(get_logger(), "机器人无法投影到避车长边");
    return std::nullopt;
  }
  const Foot chosen = (!foot_a || (foot_b && foot_b->distance < foot_a->distance)) ? *foot_b : *foot_a;
  ctx.side = chosen.side;
  ctx.foot_s = chosen.s;
  for (const auto & sample : avoidanceLongSide(*ctx.query, ctx.side).samples) {
    ctx.side_line.emplace_back(sample.x, sample.y);
  }
  RCLCPP_INFO(get_logger(), "选用避车长边 %s，垂足弧长 %.3f，距离 %.3f",
    ctx.side == garage_road_model::SideId::A ? "A" : "B", chosen.s, chosen.distance);
  RCLCPP_INFO(get_logger(), "robot: (%.3f, %.3f)", ctx.robot.x(), ctx.robot.y());
  RCLCPP_INFO(get_logger(), "car: (%.3f, %.3f)", ctx.car.x(), ctx.car.y());
  publishDebugMarkers(ctx.robot, ctx.car, ctx.side_line);
  return ctx;
}

// 停靠朝向：侧边从 start 到 end 的切线。采样 yaw 随弧长增大，就是这个方向。拐角处切线不唯一。
std::optional<double> parkingYawAlongSide(
  const garage_road_model::RoadModelQuery & query,
  garage_road_model::SideId side,
  double s)
{
  const auto pose = query.evaluateSide(side, s);
  if (pose.status != garage_road_model::QueryStatus::Ok || pose.corner) {
    return std::nullopt;
  }
  return pose.yaw;
}

// 垂足已经投好。横向仍用法线展开搜索带；搜索沿远离来车的方向，车头另取 start 到 end。
bool CarAvoidancePointServer::normalAtFoot(SearchContext & ctx)
{
  const auto pose = ctx.query->evaluateSide(ctx.side, ctx.foot_s);
  const auto normal = ctx.query->inwardNormal(ctx.side, ctx.foot_s, ctx.channel);
  if (pose.status != garage_road_model::QueryStatus::Ok || pose.corner) {
    RCLCPP_ERROR(get_logger(), "投影点没有沿侧边的切线：%s",
      pose.corner ? "拐角处切线不唯一" : pose.reason.c_str());
    return false;
  }
  if (normal.status != garage_road_model::QueryStatus::Ok) {
    RCLCPP_ERROR(get_logger(), "投影点没有朝内法线：%s", normal.reason.c_str());
    return false;
  }
  const double along_yaw = flipIfFacingCar(pose.yaw, ctx.robot, ctx.car);
  ctx.away_sign = angleDiff(along_yaw, pose.yaw) < M_PI / 2.0 ? 1.0 : -1.0;
  RCLCPP_INFO(get_logger(), "搜索沿弧长%s远离来车，车头沿侧边从 start 到 end",
    ctx.away_sign > 0.0 ? "增大" : "减小");
  return true;
}

// 先从外部停车点里选一个能用的。
std::optional<geometry_msgs::msg::PoseStamped> CarAvoidancePointServer::findExternalPoint(
  const Task & task, const SearchContext & ctx)
{
  geometry_msgs::msg::PoseArray::ConstSharedPtr stop_points;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stop_points = vehicle_avoidance_stop_points_;
  }
  if (!stop_points || stop_points->poses.empty()) {
    RCLCPP_INFO(get_logger(), "话题无有效外部停车点，使用自搜索");
    return std::nullopt;
  }

  struct Candidate
  {
    PoseStamped pose;
    double robot_distance{0.0};
  };
  std::vector<Candidate> filtered;
  for (const auto & pose : stop_points->poses) {
    const double px = pose.position.x;
    const double py = pose.position.y;
    const double robot_distance = std::hypot(px - ctx.robot.x(), py - ctx.robot.y());
    if (robot_distance > external_point_max_distance_) {
      continue;
    }
    const double car_dx = ctx.car.x() - ctx.robot.x();
    const double car_dy = ctx.car.y() - ctx.robot.y();
    const double point_dx = px - ctx.robot.x();
    const double point_dy = py - ctx.robot.y();
    const double car_local_x = car_dx * std::cos(ctx.robot_yaw) + car_dy * std::sin(ctx.robot_yaw);
    const double point_local_x = point_dx * std::cos(ctx.robot_yaw) + point_dy * std::sin(ctx.robot_yaw);
    if (car_local_x * point_local_x >= 0.0) {
      RCLCPP_INFO(get_logger(), "外部停车点 (%.2f, %.2f) 与车同侧，过滤掉", px, py);
      continue;
    }
    Candidate candidate;
    candidate.pose.header.stamp = now();
    candidate.pose.header.frame_id = "map";
    candidate.pose.pose.position.x = px;
    candidate.pose.pose.position.y = py;
    candidate.pose.pose.orientation = pose.orientation;
    candidate.robot_distance = robot_distance;
    filtered.push_back(candidate);
  }
  std::sort(filtered.begin(), filtered.end(), [](const Candidate & lhs, const Candidate & rhs) {
      return lhs.robot_distance < rhs.robot_distance;
    });
  RCLCPP_INFO(get_logger(), "外部停车点过滤后剩余: %zu个", filtered.size());
  if (filtered.empty()) {
    RCLCPP_INFO(get_logger(), "过滤以后，无有效外部停车点，使用自搜索");
    return std::nullopt;
  }

  RCLCPP_INFO(get_logger(), "尝试外部停车点，共%zu个", filtered.size());
  struct Passed
  {
    PoseStamped pose;
    double side_distance{0.0};
    double robot_distance{0.0};
  };
  std::vector<Passed> passed;
  for (auto & candidate : filtered) {
    const Vec2 point(candidate.pose.pose.position.x, candidate.pose.pose.position.y);
    const bool inside = pointInPolygon(point, task.vertices);
    if (inside) {
      const auto foot = nearestFoot(*ctx.query, ctx.side, point);
      const auto yaw = foot ? parkingYawAlongSide(*ctx.query, ctx.side, foot->s) : std::nullopt;
      if (!yaw) {
        RCLCPP_INFO(get_logger(), "外部停车点(%.2f, %.2f) 在通道内，但该处没有沿侧边的切线，跳过",
          point.x(), point.y());
        continue;
      }
      candidate.pose.pose.orientation = yawToQuaternion(*yaw);
    } else {
      RCLCPP_INFO(get_logger(), "外部停车点(%.2f, %.2f) 在通道外，将跳过服务检查", point.x(), point.y());
    }
    if (!validateCandidate(task, ctx, candidate.pose, !inside)) {
      continue;
    }
    double side_distance = std::numeric_limits<double>::infinity();
    for (size_t i = 1; i < ctx.side_line.size(); ++i) {
      side_distance = std::min(
        side_distance, distancePointToSegment(point, ctx.side_line[i - 1], ctx.side_line[i]));
    }
    passed.push_back(Passed{candidate.pose, side_distance, candidate.robot_distance});
  }
  if (passed.empty()) {
    RCLCPP_INFO(get_logger(), "有外部停车点，但是所有外部停车点在筛选判断以后均不满足，回退自搜索");
    return std::nullopt;
  }

  const auto best = std::min_element(passed.begin(), passed.end(), [](const Passed & lhs, const Passed & rhs) {
      if (std::abs(lhs.side_distance - rhs.side_distance) > 1e-6) {
        return lhs.side_distance < rhs.side_distance;
      }
      return lhs.robot_distance < rhs.robot_distance;
    });
  PoseStamped result = best->pose;
  const bool inside = pointInPolygon(Vec2(result.pose.position.x, result.pose.position.y), task.vertices);
  if (inside) {
    result.pose.position.z = 20.0;
    RCLCPP_INFO(get_logger(),
      "外部停车点通过校验共%zu个，选择最贴近长边: (%.2f, %.2f)，该点在通道内，使用避让方向，z = 20.0",
      passed.size(), result.pose.position.x, result.pose.position.y);
  } else {
    result.pose.position.z = 30.0;
    RCLCPP_INFO(get_logger(),
      "外部停车点通过校验共%zu个，选择最贴近长边: (%.2f, %.2f)，该点在通道外，方向保持不变，z = 30.0",
      passed.size(), result.pose.position.x, result.pose.position.y);
  }
  return result;
}

// 沿所选边生成候选点，搜索框，返回第一个通过检查的。
std::optional<geometry_msgs::msg::PoseStamped> CarAvoidancePointServer::findAvoidancePoint(
  const Task & task, const SearchContext & ctx)
{
  const auto facing_yaw = parkingYawAlongSide(*ctx.query, ctx.side, ctx.foot_s);
  if (!facing_yaw) {
    RCLCPP_ERROR(get_logger(), "机器人垂足处没有沿侧边的切线，无法判断前后");
    return std::nullopt;
  }
  const double retreat_yaw = ctx.away_sign < 0.0 ? addAngles(*facing_yaw, M_PI) : *facing_yaw;
  const bool ahead = angleDiff(retreat_yaw, ctx.robot_yaw) < M_PI / 2.0;
  const double extra = ahead ? search_radius_extra_dis_ : 0.0;
  const double near_distance = search_radius_min_ + extra;
  const double far_distance = search_radius_max_ + extra;
  if (ahead) {
    RCLCPP_INFO(get_logger(), "在机器人前方避车, 额外增加%.3f米搜索距离", search_radius_extra_dis_);
  } else {
    RCLCPP_INFO(get_logger(), "在机器人后方避车");
  }

  const double interval = std::max(search_point_interval_, 1e-3);
  const double lateral_lo = std::min(outside_min_, outside_max_);
  const double lateral_hi = std::max(outside_min_, outside_max_);
  const Vec2 interior = shortEdgeInteriorPoint(ctx.channel);
  std::vector<PoseStamped> candidates;
  Ring search_band;

  const auto foot = ctx.query->evaluateSide(ctx.side, ctx.foot_s);
  if (foot.status != garage_road_model::QueryStatus::Ok) {
    RCLCPP_ERROR(get_logger(), "避车长边垂足处没有切线，无法生成搜索框");
    return std::nullopt;
  }
  const double along_yaw = flipIfFacingCar(foot.yaw, ctx.robot, ctx.car);
  const double away_sign = angleDiff(along_yaw, foot.yaw) < M_PI / 2.0 ? 1.0 : -1.0;
  const auto & boundary = avoidanceLongSide(*ctx.query, ctx.side);
  const double side_begin = boundary.samples.front().s;
  const double side_end = boundary.samples.back().s;
  const double s_front = ctx.foot_s + away_sign * far_distance;
  if (s_front < side_begin - 1e-6 || s_front > side_end + 1e-6) {
    RCLCPP_ERROR(get_logger(), "搜索框构造失败：搜索框最前面越过了通道前方端口");
    return std::nullopt;
  }
  const auto front_pose = ctx.query->evaluateSide(ctx.side, s_front);
  if (front_pose.status == garage_road_model::QueryStatus::Ok) {
    const Vec2 left(-std::sin(front_pose.yaw), std::cos(front_pose.yaw));
    Vec2 outward = Vec2::Zero();
    if (pointInPolygon(front_pose.position + 0.15 * left, ctx.channel)) {
      outward = -left;
    } else if (pointInPolygon(front_pose.position - 0.15 * left, ctx.channel)) {
      outward = left;
    }
    const auto & cap = away_sign > 0.0 ? ctx.query->model().end_cap : ctx.query->model().start_cap;
    if (outward.squaredNorm() > 0.5 && cap.size() >= 2) {
      const Edge cap_edge{cap.front(), cap.back()};
      const Vec2 inner = front_pose.position + outward * lateral_lo;
      const Vec2 outer = front_pose.position + outward * lateral_hi;
      if (isPointBeyondEdge(inner, cap_edge, interior) || isPointBeyondEdge(outer, cap_edge, interior)) {
        RCLCPP_ERROR(get_logger(), "搜索框构造失败：搜索框最前面越过了通道前方端口");
        return std::nullopt;
      }
    }
  }

  const double span = far_distance - near_distance;
  const int along_steps = std::max(0, static_cast<int>(span / interval));
  const int lateral_steps = std::max(0, static_cast<int>((lateral_hi - lateral_lo) / interval));
  Ring inner_edge;
  Ring outer_edge;
  for (int i = 0; i <= along_steps; ++i) {
    const double station = ctx.foot_s + away_sign * (near_distance + interval * i);
    if (station < side_begin - 1e-6 || station > side_end + 1e-6) {
      continue;
    }
    const auto pose = ctx.query->evaluateSide(ctx.side, station);
    const auto normal = ctx.query->inwardNormal(ctx.side, station, ctx.channel);
    if (pose.status != garage_road_model::QueryStatus::Ok || pose.corner ||
      normal.status != garage_road_model::QueryStatus::Ok)
    {
      continue;
    }
    const double park_yaw = pose.yaw;
    const Vec2 outward = -normal.direction;
    inner_edge.push_back(pose.position + outward * lateral_lo);
    outer_edge.push_back(pose.position + outward * lateral_hi);
    for (int j = 0; j <= lateral_steps; ++j) {
      const double lateral = lateral_lo + interval * j;
      if (lateral > lateral_hi + 1e-9) {
        continue;
      }
      const Vec2 point = pose.position + outward * lateral;
      PoseStamped candidate;
      candidate.header.stamp = now();
      candidate.header.frame_id = "map";
      candidate.pose.position.x = point.x();
      candidate.pose.position.y = point.y();
      candidate.pose.orientation = yawToQuaternion(park_yaw);
      candidates.push_back(candidate);
    }
  }
  search_band = inner_edge;
  search_band.insert(search_band.end(), outer_edge.rbegin(), outer_edge.rend());
  std::sort(candidates.begin(), candidates.end(), [&](const PoseStamped & lhs, const PoseStamped & rhs) {
      const double left = std::hypot(lhs.pose.position.x - ctx.robot.x(), lhs.pose.position.y - ctx.robot.y());
      const double right = std::hypot(rhs.pose.position.x - ctx.robot.x(), rhs.pose.position.y - ctx.robot.y());
      return left < right;
    });

  if (!search_band.empty()) {
    publishDebugMarkers(ctx.robot, ctx.car, ctx.side_line, &search_band);
  }
  RCLCPP_INFO(get_logger(), "一共%zu个候选避让点", candidates.size());
  for (const auto & candidate : candidates) {
    if (validateCandidate(task, ctx, candidate, false)) {
      PoseStamped result = candidate;
      result.pose.position.z = 10.0;
      publishDebugMarkers(ctx.robot, ctx.car, ctx.side_line, search_band.empty() ? nullptr : &search_band, &result);
      return result;
    }
  }
  RCLCPP_INFO(get_logger(), "当前搜索区域没有找到避让点");
  return std::nullopt;
}

}  // namespace find_free_space
