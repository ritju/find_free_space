#include "find_free_space/car_avoidance_point_server.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

#include "geometry_msgs/msg/point.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"

namespace find_free_space
{

namespace
{

constexpr uint8_t kMaxFreeCost = 253;

std::vector<cv::Point2f> toContour(const geometry_msgs::msg::Polygon & polygon)
{
  std::vector<cv::Point2f> contour;
  contour.reserve(polygon.points.size());
  for (const auto & point : polygon.points) {
    contour.emplace_back(point.x, point.y);
  }
  return contour;
}

}  // namespace

bool CarAvoidancePointServer::validateCandidate(
  const Task & task,
   const SearchContext & ctx,
    const PoseStamped & candidate,
  bool skip_service_check)
{
  const MapView & map = ctx.map;
  const double x = candidate.pose.position.x;
  const double y = candidate.pose.position.y;

  auto local_x = [&](double world_x, double world_y) {
      const double dx = world_x - ctx.robot.x();
      const double dy = world_y - ctx.robot.y();
      return dx * std::cos(ctx.robot_yaw) + dy * std::sin(ctx.robot_yaw);
    };
  const double car_local_x = local_x(ctx.car.x(), ctx.car.y());
  const double point_local_x = local_x(x, y);
  if (car_local_x * point_local_x >= 0.0) {
    RCLCPP_INFO(get_logger(),
      "候选点(%.2f,%.2f) 与车在机器人同侧(car_local_x=%.2f, point_local_x=%.2f)，拒绝",
      x, y, car_local_x, point_local_x);
    return false;
  }

  // 点周围无障碍
  const int point_x_p = static_cast<int>(std::clamp((x - map.origin_x) / map.resolution, 0.0, map.width - 1.0));
  const int point_y_p = static_cast<int>(std::clamp((y - map.origin_y) / map.resolution, 0.0, map.height - 1.0));
  const int point_free_radius_pixels = std::max(1, static_cast<int>(point_free_check_radius_ / map.resolution));
  if (!checkPointIsFree(map, point_x_p, point_y_p, point_free_radius_pixels)) {
    RCLCPP_INFO(get_logger(), "候选点(%.2f,%.2f) 周围%.1fm内有障碍", x, y, point_free_check_radius_);
    return false;
  }

  // 不在禁扫区
  if (isPointInSpecialTerrain(x, y)) {
    RCLCPP_INFO(get_logger(), "候选点(%.2f,%.2f) 在禁扫区内", x, y);
    return false;
  }

  // footprint 不与禁扫区重叠
  const double pose_yaw = tf2::getYaw(candidate.pose.orientation);
  const Ring local_footprint = footprint();
  if (isFootprintInSpecialTerrain(local_footprint, x, y, pose_yaw)) {
    RCLCPP_INFO(get_logger(), "候选点(%.2f,%.2f) footprint与禁扫区重叠", x, y);
    return false;
  }

  const int robot_x_p = static_cast<int>(
    std::clamp((ctx.robot.x() - map.origin_x) / map.resolution, 0.0, map.width - 1.0));
  const int robot_y_p = static_cast<int>(
    std::clamp((ctx.robot.y() - map.origin_y) / map.resolution, 0.0, map.height - 1.0));
  if (!checkPathIsFree(map, robot_x_p, robot_y_p, point_x_p, point_y_p)) {
    RCLCPP_INFO(get_logger(), "候选点(%.2f,%.2f) 连线有障碍", x, y);
    return false;
  }

  if (!local_footprint.empty() && !checkFootprintAtPose(ctx, local_footprint, x, y, pose_yaw)) {
    RCLCPP_INFO(get_logger(), "候选点(%.2f,%.2f) footprint撞障碍", x, y);
    return false;
  }

  if (sweepEnabled(local_footprint) &&
    !checkFootprintSweep(map, local_footprint, ctx.robot, Vec2(x, y), pose_yaw))
  {
    RCLCPP_INFO(get_logger(), "候选点(%.2f,%.2f) 扫掠区域有障碍", x, y);
    return false;
  }

  // 外部停车点且在通道外：跳过服务检查，直接通过
  if (skip_service_check) {
    RCLCPP_INFO(get_logger(), "候选点(%.2f,%.2f) 为通道外外部停车点，跳过 /check_car_passable 服务检查，直接通过", x, y);
    return true;
  }

  auto request = std::make_shared<IsCarPassable::Request>();
  request->robot_pose = candidate;
  request->car_pose = task.goal->car_pose;
  request->size = task.goal->car_size;
  request->find_parking_point = true;
  const auto start_time = std::chrono::steady_clock::now();
  const bool check_result = checkAvoidance(request);
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  RCLCPP_INFO(get_logger(), "候选点(%.2f,%.2f) 服务检查 result=%d, time=%.3fs", x, y, check_result, elapsed);
  if (check_result && elapsed < check_service_max_time_) {
    return true;
  }
  RCLCPP_INFO(get_logger(), "候选点(%.2f,%.2f) 服务检查不通过", x, y);
  return false;
}

bool CarAvoidancePointServer::checkPointIsFree(const MapView & map, int center_x, int center_y, int radius) const
{
  for (int y = std::max(0, center_y - radius); y < std::min(map.height, center_y + radius + 1); ++y) {
    for (int x = std::max(0, center_x - radius); x < std::min(map.width, center_x + radius + 1); ++x) {
      const double distance = std::hypot(x - center_x, y - center_y);
      if (distance <= radius && map.at(x, y) == 254) {
        return false;
      }
    }
  }
  return true;
}

bool CarAvoidancePointServer::isPointInSpecialTerrain(double x, double y) const
{
  std::shared_ptr<const std::vector<geometry_msgs::msg::Polygon>> polygons;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    polygons = special_terrain_polygons_;
  }
  if (!polygons) {
    return false;
  }
  for (const auto & polygon : *polygons) {
    if (polygon.points.size() < 3) {
      continue;
    }
    if (cv::pointPolygonTest(toContour(polygon), cv::Point2f(x, y), false) >= 0) {
      return true;
    }
  }
  return false;
}

bool CarAvoidancePointServer::isFootprintInSpecialTerrain(
  const Ring & local_footprint, double x, double y, double yaw) const
{
  std::shared_ptr<const std::vector<geometry_msgs::msg::Polygon>> polygons;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    polygons = special_terrain_polygons_;
  }
  if (!polygons || local_footprint.empty()) {
    return false;
  }
  const Ring fp_world = transformFootprint(local_footprint, x, y, yaw);
  std::vector<cv::Point2f> fp_contour;
  for (const auto & point : fp_world) {
    fp_contour.emplace_back(point.x(), point.y());
  }
  for (const auto & polygon : *polygons) {
    if (polygon.points.size() < 3) {
      continue;
    }
    const auto poly_pts = toContour(polygon);
    for (const auto & point : fp_contour) {
      if (cv::pointPolygonTest(poly_pts, point, false) >= 0) {
        return true;
      }
    }
    for (const auto & point : poly_pts) {
      if (cv::pointPolygonTest(fp_contour, point, false) >= 0) {
        return true;
      }
    }
    std::vector<cv::Point2f> intersection;
    if (cv::intersectConvexConvex(fp_contour, poly_pts, intersection) > 0) {
      return true;
    }
  }
  return false;
}

bool CarAvoidancePointServer::polygonCellsFree(const MapView & map, const Ring & polygon) const
{
  if (polygon.empty()) {
    return true;
  }
  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  for (const auto & point : polygon) {
    min_x = std::min(min_x, point.x());
    max_x = std::max(max_x, point.x());
    min_y = std::min(min_y, point.y());
    max_y = std::max(max_y, point.y());
  }
  auto to_cell = [](double value, double origin, double resolution, int size) {
      return std::clamp(static_cast<int>(std::floor((value - origin) / resolution)), 0, size - 1);
    };
  const int x0 = to_cell(min_x, map.origin_x, map.resolution, map.width);
  const int x1 = to_cell(max_x, map.origin_x, map.resolution, map.width);
  const int y0 = to_cell(min_y, map.origin_y, map.resolution, map.height);
  const int y1 = to_cell(max_y, map.origin_y, map.resolution, map.height);
  for (int cell_y = y0; cell_y <= y1; ++cell_y) {
    for (int cell_x = x0; cell_x <= x1; ++cell_x) {
      if (map.at(cell_x, cell_y) <= kMaxFreeCost) {
        continue;
      }
      const double center_x = map.origin_x + (cell_x + 0.5) * map.resolution;
      const double center_y = map.origin_y + (cell_y + 0.5) * map.resolution;
      if (cellIntersectsConvex(center_x, center_y, map.resolution, polygon)) {
        return false;
      }
    }
  }
  return true;
}

bool CarAvoidancePointServer::checkFootprintAtPose(
  const SearchContext & ctx, const Ring & local_footprint, double x, double y, double yaw) const
{
  if (!ctx.nav_costmap || local_footprint.empty()) {
    return local_footprint.empty();
  }
  std::vector<geometry_msgs::msg::Point> footprint;
  footprint.reserve(local_footprint.size());
  for (const auto & point : local_footprint) {
    geometry_msgs::msg::Point corner;
    corner.x = point.x();
    corner.y = point.y();
    footprint.push_back(corner);
  }
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(ctx.nav_costmap.get());
  const double boundary_cost = checker.footprintCostAtPose(x, y, yaw, footprint);
  if (boundary_cost < 0.0 || boundary_cost > static_cast<double>(kMaxFreeCost)) {
    return false;
  }
  return polygonCellsFree(ctx.map, transformFootprint(local_footprint, x, y, yaw));
}

bool CarAvoidancePointServer::sweepEnabled(const Ring & local_footprint) const
{
  if (local_footprint.size() < 2) {
    return false;
  }
  double longest = 0.0;
  for (size_t i = 0; i < local_footprint.size(); ++i) {
    longest = std::max(longest, (local_footprint[(i + 1) % local_footprint.size()] - local_footprint[i]).norm());
  }
  return footprint_sweep_long_step_ <= longest;
}

bool CarAvoidancePointServer::checkPathIsFree(
  const MapView & map, int current_x, int current_y, int target_x, int target_y) const
{
  // Bresenham 取两点连线上的所有像素
  const int dx = std::abs(target_x - current_x);
  const int dy = std::abs(target_y - current_y);
  const int sx = current_x < target_x ? 1 : -1;
  const int sy = current_y < target_y ? 1 : -1;
  int err = dx - dy;
  int x = current_x;
  int y = current_y;
  while (true) {
    if (x >= 0 && x < map.width && y >= 0 && y < map.height && map.at(x, y) > kMaxFreeCost) {
      return false;
    }
    if (x == target_x && y == target_y) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }
    if (e2 < dx) {
      err += dx;
      y += sy;
    }
  }
  return true;
}

bool CarAvoidancePointServer::checkFootprintSweep(
  const MapView & map, const Ring & local_footprint, const Vec2 & robot, const Vec2 & target,
  double target_yaw) const
{
  if (local_footprint.empty()) {
    return true;
  }
  // 两端都用目标朝向，只有平移，两个车身的凸包就是扫过的区域
  Ring corners = transformFootprint(local_footprint, robot.x(), robot.y(), target_yaw);
  const Ring target_corners = transformFootprint(local_footprint, target.x(), target.y(), target_yaw);
  corners.insert(corners.end(), target_corners.begin(), target_corners.end());
  return polygonCellsFree(map, convexHull(std::move(corners)));
}

bool CarAvoidancePointServer::checkAvoidance(const IsCarPassable::Request::SharedPtr request)
{
  while (!check_avoidance_service_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_INFO(get_logger(), "服务未就绪，等待中...");
  }
  auto future = check_avoidance_service_->async_send_request(request);
  future.wait();
  return future.get()->is_car_passable;
}

}  // namespace find_free_space
