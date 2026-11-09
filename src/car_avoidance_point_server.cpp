#include "find_free_space/car_avoidance_point_server.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "tf2/exceptions.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"

namespace find_free_space
{

namespace
{

std::string formatRing(const Ring & ring)
{
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(3);
  for (size_t i = 0; i < ring.size(); ++i) {
    out << (i == 0 ? "" : ", ") << "(" << ring[i].x() << ", " << ring[i].y() << ")";
  }
  return out.str();
}

}  // namespace

CarAvoidancePointServer::CarAvoidancePointServer()
: rclcpp::Node("find_free_space_action_server")
{
  RCLCPP_INFO(get_logger(), "find_free_space_action_server started.");
  initParams();

  const auto marker_qos = rclcpp::QoS(1).transient_local();
  marker_car_pose_publisher_ = create_publisher<visualization_msgs::msg::Marker>("marker_car_pose", marker_qos);
  marker_robot_pose_publisher_ = create_publisher<visualization_msgs::msg::Marker>("marker_robot_pose", marker_qos);
  marker_avoidance_side_publisher_ =
    create_publisher<visualization_msgs::msg::Marker>("marker_avoidance_side", marker_qos);
  marker_searching_rect_publisher_ =
    create_publisher<visualization_msgs::msg::Marker>("marker_searching_rect", marker_qos);
  marker_parking_point_publisher_ =
    create_publisher<visualization_msgs::msg::Marker>("marker_parking_point", marker_qos);
  marker_special_terrain_publisher_ =
    create_publisher<visualization_msgs::msg::Marker>("marker_special_terrain", marker_qos);
  marker_all_passages_publisher_ =
    create_publisher<visualization_msgs::msg::Marker>("marker_all_passages", marker_qos);

  action_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  costmap_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  service_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  footprint_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions footprint_options;
  footprint_options.callback_group = footprint_cb_group_;
  footprint_sub_ = create_subscription<geometry_msgs::msg::PolygonStamped>(
    topic_name_footprint_, rclcpp::QoS(1),
    std::bind(&CarAvoidancePointServer::footprintCallback, this, std::placeholders::_1),
    footprint_options);

  rcl_action_server_options_t action_options = rcl_action_server_get_default_options();
  action_options.feedback_topic_qos.depth = 1;
  action_options.feedback_topic_qos.durability = RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL;
  action_server_ = rclcpp_action::create_server<FindCarAvoidancePoint>(
    this, "/find_car_avoidance_point_action",
    [](const rclcpp_action::GoalUUID &, std::shared_ptr<const FindCarAvoidancePoint::Goal>) {
      return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    },
    [](const std::shared_ptr<GoalHandle>) {
      return rclcpp_action::CancelResponse::REJECT;
    },
    [this](const std::shared_ptr<GoalHandle> goal_handle) {
      std::thread{[this, goal_handle]() {execute(goal_handle);}}.detach();
    },
    action_options, action_cb_group_);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);

  check_avoidance_service_ = create_client<IsCarPassable>(
    "/check_car_passable", rmw_qos_profile_services_default, service_cb_group_);

  rclcpp::SubscriptionOptions road_model_options;
  road_model_options.callback_group = costmap_cb_group_;
  road_models_sub_ = create_subscription<garage_road_model::msg::RoadModelArray>(
    "/garage_road_models", rclcpp::QoS(1).reliable().transient_local(),
    std::bind(&CarAvoidancePointServer::roadModelsCallback, this, std::placeholders::_1),
    road_model_options);
}

void CarAvoidancePointServer::initParams()
{
  topic_name_global_costmap_ = declare_parameter<std::string>("topic_name_global_costmap", "");
  topic_name_footprint_ = declare_parameter<std::string>("topic_name_footprint", "");
  search_radius_min_ = declare_parameter("search_radius_min", 3.0);
  search_radius_max_ = declare_parameter("search_radius_max", 4.0);
  search_radius_extra_dis_ = declare_parameter("search_radius_extra_dis", 2.5);
  search_point_interval_ = declare_parameter("search_point_interval", 0.15);
  outside_min_ = declare_parameter("outside_min", 0.0);
  outside_max_ = declare_parameter("outside_max", 0.5);
  check_service_max_time_ = declare_parameter("check_service_max_time", 0.5);
  show_global_costmap_raw_cv2_ = declare_parameter("show_global_costmap_raw_cv2", false);
  show_global_costmap_raw_colored_cv2_ = declare_parameter("show_global_costmap_raw_colored_cv2", false);
  external_point_max_distance_ = declare_parameter("external_point_max_distance", 8.0);
  point_free_check_radius_ = declare_parameter("point_free_check_radius", 0.2);
  footprint_sweep_long_step_ = declare_parameter("footprint_sweep_long_step", 999.0);
  footprint_sweep_short_step_ = declare_parameter("footprint_sweep_short_step", 999.0);
  max_robot_pose_age_ = declare_parameter("max_robot_pose_age", 1.0);
  robot_pose_wait_timeout_ = declare_parameter("robot_pose_wait_timeout", 1.5);

  RCLCPP_INFO(get_logger(), "topic_name_global_costmap: %s", topic_name_global_costmap_.c_str());
  RCLCPP_INFO(get_logger(), "topic_name_footprint: %s", topic_name_footprint_.c_str());
  RCLCPP_INFO(get_logger(), "search_radius_min: %f", search_radius_min_);
  RCLCPP_INFO(get_logger(), "search_radius_max: %f", search_radius_max_);
  RCLCPP_INFO(get_logger(), "search_radius_extra_dis: %f", search_radius_extra_dis_);
  RCLCPP_INFO(get_logger(), "search_point_interval: %f", search_point_interval_);
  RCLCPP_INFO(get_logger(), "outside_min: %f", outside_min_);
  RCLCPP_INFO(get_logger(), "outside_max: %f", outside_max_);
  RCLCPP_INFO(get_logger(), "check_service_max_time: %f", check_service_max_time_);
  RCLCPP_INFO(get_logger(), "show_global_costmap_raw_cv2: %d", show_global_costmap_raw_cv2_);
  RCLCPP_INFO(get_logger(), "show_global_costmap_raw_colored_cv2: %d", show_global_costmap_raw_colored_cv2_);
  RCLCPP_INFO(get_logger(), "external_point_max_distance: %f", external_point_max_distance_);
  RCLCPP_INFO(get_logger(), "point_free_check_radius: %f", point_free_check_radius_);
  RCLCPP_INFO(get_logger(), "footprint_sweep_long_step: %f", footprint_sweep_long_step_);
  RCLCPP_INFO(get_logger(), "footprint_sweep_short_step: %f", footprint_sweep_short_step_);
  RCLCPP_INFO(get_logger(), "max_robot_pose_age: %f", max_robot_pose_age_);
  RCLCPP_INFO(get_logger(), "robot_pose_wait_timeout: %f", robot_pose_wait_timeout_);
}

void CarAvoidancePointServer::footprintCallback(const geometry_msgs::msg::PolygonStamped::ConstSharedPtr msg)
{
  if (!footprint_sub_) {
    return;
  }
  const auto & points = msg->polygon.points;
  std::vector<double> edges_distance;
  for (size_t i = 0; i < 4; ++i) {
    const auto & a = points.at(i);
    const auto & b = points.at((i + 1) % 4);
    const double length = std::hypot(static_cast<double>(a.x) - b.x, static_cast<double>(a.y) - b.y);
    edges_distance.push_back(std::nearbyint(length * 100.0) / 100.0);
  }
  const double robot_width = *std::min_element(edges_distance.begin(), edges_distance.end());
  // 用边长构造以机器人中心为原点的局部 footprint
  const double half_long = *std::max_element(edges_distance.begin(), edges_distance.end()) / 2.0;
  const double half_short = robot_width / 2.0;
  const Ring local = {
    {half_long, half_short},
    {half_long, -half_short},
    {-half_long, -half_short},
    {-half_long, half_short}};
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    footprint_vertices_ = local;
  }
  RCLCPP_INFO(get_logger(), "robot_width: %.3f, footprint_vertices (local): [%s]",
    robot_width, formatRing(local).c_str());
  footprint_sub_.reset();
}

void CarAvoidancePointServer::globalCostmapCallback(const nav2_msgs::msg::Costmap::ConstSharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    global_costmap_ = msg;
  }
  if (!show_global_costmap_raw_colored_cv2_ && !show_global_costmap_raw_cv2_) {
    return;
  }
  const int width = static_cast<int>(msg->metadata.size_x);
  const int height = static_cast<int>(msg->metadata.size_y);
  const cv::Mat costmap_data(height, width, CV_8UC1, const_cast<uint8_t *>(msg->data.data()));
  RCLCPP_INFO_ONCE(get_logger(), "count0: %d", cv::countNonZero(costmap_data == 0));
  RCLCPP_INFO_ONCE(get_logger(), "count254: %d", cv::countNonZero(costmap_data == 254));
  RCLCPP_INFO_ONCE(get_logger(), "count255: %d", cv::countNonZero(costmap_data == 255));
  if (show_global_costmap_raw_colored_cv2_) {
    cv::Mat colored_map;
    cv::applyColorMap(costmap_data, colored_map, cv::COLORMAP_JET);
    cv::imshow("Global Costmap Raw Colored", colored_map);
  }
  if (show_global_costmap_raw_cv2_) {
    cv::imshow("Global Costmap Raw", costmap_data);
  }
  cv::waitKey(1);
}

void CarAvoidancePointServer::specialTerrainCallback(const garage_utils_msgs::msg::Polygons::ConstSharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    special_terrain_polygons_ = std::make_shared<const std::vector<geometry_msgs::msg::Polygon>>(msg->polygons);
  }
  for (size_t idx = 0; idx < msg->polygons.size(); ++idx) {
    Ring ring;
    for (const auto & p : msg->polygons[idx].points) {
      ring.emplace_back(p.x, p.y);
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 20000, "禁扫区[%zu]: %s", idx, formatRing(ring).c_str());
  }
  RCLCPP_INFO_ONCE(get_logger(), "收到禁扫区: %zu个区域", msg->polygons.size());

  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = now();
  marker.id = 10;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.scale.x = 0.05;
  if (msg->polygons.empty()) {
    marker.action = visualization_msgs::msg::Marker::DELETE;
    marker_special_terrain_publisher_->publish(marker);
    return;
  }
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 0.5;
  for (const auto & polygon : msg->polygons) {
    const auto & pts = polygon.points;
    if (pts.size() < 2) {
      continue;
    }
    for (size_t i = 0; i < pts.size(); ++i) {
      geometry_msgs::msg::Point p1;
      p1.x = pts[i].x;
      p1.y = pts[i].y;
      p1.z = pts[i].z;
      geometry_msgs::msg::Point p2;
      p2.x = pts[(i + 1) % pts.size()].x;
      p2.y = pts[(i + 1) % pts.size()].y;
      p2.z = pts[(i + 1) % pts.size()].z;
      marker.points.push_back(p1);
      marker.points.push_back(p2);
    }
  }
  marker_special_terrain_publisher_->publish(marker);
}

void CarAvoidancePointServer::vehicleStopPointsCallback(const geometry_msgs::msg::PoseArray::ConstSharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    vehicle_avoidance_stop_points_ = msg;
  }
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "收到避车停车点: %zu个", msg->poses.size());
  if (!msg->poses.empty()) {
    const auto & orientation = msg->poses[0].orientation;
    const bool is_default_orientation =
      std::abs(orientation.x) < 1e-6 && std::abs(orientation.y) < 1e-6 &&
      std::abs(orientation.z) < 1e-6 && std::abs(std::abs(orientation.w) - 1.0) < 1e-6;
    if (is_default_orientation) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "外部停车点没有发 yaw，将使用避让方向自行计算");
    }
  }
}

void CarAvoidancePointServer::roadModelsCallback(
  const garage_road_model::msg::RoadModelArray::ConstSharedPtr msg)
{
  std::vector<garage_road_model::RoadModelData> roads;
  roads.reserve(msg->roads.size());
  auto load_points = [](const geometry_msgs::msg::Polygon & polygon) {
      std::vector<Eigen::Vector2d> points;
      points.reserve(polygon.points.size());
      for (const auto & point : polygon.points) {
        points.emplace_back(point.x, point.y);
      }
      return points;
    };
  for (const auto & item : msg->roads) {
    garage_road_model::RoadModelData road;
    road.source_index = item.source_index;
    road.boundary = load_points(item.boundary);
    road.start_cap = load_points(item.start_cap);
    road.end_cap = load_points(item.end_cap);
    road.valid = item.valid;
    road.stage = item.stage;
    road.reason = item.reason;
    auto load_side = [&](const garage_road_model::msg::SideBoundary & side) {
        garage_road_model::SideBoundary out;
        out.raw = load_points(side.raw);
        out.length = side.length;
        for (const auto & sample : side.samples) {
          garage_road_model::RoadSampleData data;
          data.x = sample.x;
          data.y = sample.y;
          data.yaw = sample.yaw;
          data.s = sample.s;
          data.corner = sample.corner;
          out.samples.push_back(data);
        }
        return out;
      };
    road.side_a = load_side(item.side_a);
    road.side_b = load_side(item.side_b);
    roads.push_back(std::move(road));
  }
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    cached_road_models_ = std::move(roads);
  }
  RCLCPP_INFO(get_logger(), "收到 /garage_road_models，revision %lu，数量 %zu",
    static_cast<unsigned long>(msg->revision), msg->roads.size());
}

namespace
{

std::vector<Eigen::Vector2d> cleanRing(const std::vector<Eigen::Vector2d> & points)
{
  std::vector<Eigen::Vector2d> ring;
  for (const auto & point : points) {
    if (ring.empty() || (ring.back() - point).norm() > 1e-4) {
      ring.push_back(point);
    }
  }
  if (ring.size() >= 2 && (ring.front() - ring.back()).norm() <= 1e-4) {
    ring.pop_back();
  }
  return ring;
}

double ringMismatch(const std::vector<Eigen::Vector2d> & lhs, std::vector<Eigen::Vector2d> rhs)
{
  if (lhs.size() < 3 || lhs.size() != rhs.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double best = std::numeric_limits<double>::infinity();
  for (int direction = 0; direction < 2; ++direction) {
    const int count = static_cast<int>(lhs.size());
    for (int shift = 0; shift < count; ++shift) {
      double worst = 0.0;
      for (int i = 0; i < count; ++i) {
        worst = std::max(
          worst, (lhs[static_cast<size_t>(i)] - rhs[static_cast<size_t>((i + shift) % count)]).norm());
      }
      best = std::min(best, worst);
    }
    std::reverse(rhs.begin(), rhs.end());
  }
  return best;
}

}  // namespace

std::optional<garage_road_model::RoadModelData> CarAvoidancePointServer::matchRoadModel(
  const Ring & polygon, const std::vector<garage_road_model::RoadModelData> & models,
  std::string & reason) const
{
  const auto ring = cleanRing(polygon);
  constexpr double kTolerance = 0.05;
  std::vector<size_t> matches;
  for (size_t i = 0; i < models.size(); ++i) {
    if (ringMismatch(ring, cleanRing(models[i].boundary)) <= kTolerance) {
      matches.push_back(i);
    }
  }
  if (matches.empty()) {
    reason = "当前通道没有几何对应的道路模型";
    return std::nullopt;
  }
  if (matches.size() > 1) {
    reason = "当前通道对应多个道路模型，无法确定";
    return std::nullopt;
  }
  return models[matches.front()];
}

void CarAvoidancePointServer::startTaskSubscriptions()
{
  rclcpp::SubscriptionOptions costmap_options;
  costmap_options.callback_group = costmap_cb_group_;
  global_costmap_sub_ = create_subscription<nav2_msgs::msg::Costmap>(
    topic_name_global_costmap_, rclcpp::QoS(10),
    std::bind(&CarAvoidancePointServer::globalCostmapCallback, this, std::placeholders::_1),
    costmap_options);
  special_terrain_sub_ = create_subscription<garage_utils_msgs::msg::Polygons>(
    "/cleaning_tool_retraction_areas", rclcpp::QoS(1).transient_local(),
    std::bind(&CarAvoidancePointServer::specialTerrainCallback, this, std::placeholders::_1));
  vehicle_stop_points_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
    "/vehicle_avoidance_stop_points", rclcpp::QoS(1).transient_local().reliable(),
    std::bind(&CarAvoidancePointServer::vehicleStopPointsCallback, this, std::placeholders::_1));
}

void CarAvoidancePointServer::stopTaskSubscriptions()
{
  global_costmap_sub_.reset();
  special_terrain_sub_.reset();
  vehicle_stop_points_sub_.reset();
  std::lock_guard<std::mutex> lock(data_mutex_);
  global_costmap_.reset();
  special_terrain_polygons_.reset();
  vehicle_avoidance_stop_points_.reset();
}

bool CarAvoidancePointServer::waitForGlobalCostmap(double timeout_sec)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (global_costmap_) {
        return true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::lock_guard<std::mutex> lock(data_mutex_);
  return global_costmap_ != nullptr;
}

bool CarAvoidancePointServer::snapshotCostmap(SearchContext & ctx)
{
  nav2_msgs::msg::Costmap::ConstSharedPtr costmap;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    costmap = global_costmap_;
  }
  if (!costmap || costmap->metadata.size_x == 0 || costmap->metadata.size_y == 0 ||
    costmap->data.size() < static_cast<size_t>(costmap->metadata.size_x) * costmap->metadata.size_y)
  {
    RCLCPP_ERROR(get_logger(), "全局代价地图未收到");
    return false;
  }

  ctx.costmap = costmap;
  ctx.map = MapView{
    costmap->data.data(),
    costmap->metadata.origin.position.x,
    costmap->metadata.origin.position.y,
    costmap->metadata.resolution,
    static_cast<int>(costmap->metadata.size_x),
    static_cast<int>(costmap->metadata.size_y)};
  ctx.nav_costmap = std::make_shared<nav2_costmap_2d::Costmap2D>(
    costmap->metadata.size_x, costmap->metadata.size_y, costmap->metadata.resolution,
    costmap->metadata.origin.position.x, costmap->metadata.origin.position.y);
  for (unsigned int y = 0; y < costmap->metadata.size_y; ++y) {
    for (unsigned int x = 0; x < costmap->metadata.size_x; ++x) {
      ctx.nav_costmap->setCost(x, y, ctx.map.at(static_cast<int>(x), static_cast<int>(y)));
    }
  }
  return true;
}

void CarAvoidancePointServer::waitForSpecialTerrain(double timeout_sec)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (special_terrain_polygons_) {
        return;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (!special_terrain_polygons_) {
    RCLCPP_WARN(get_logger(), "等待禁扫区超时，本次任务不拦截禁扫区");
  }
}

std::optional<CarAvoidancePointServer::PoseStamped> CarAvoidancePointServer::getFreshRobotPose()
{
  const auto deadline = now() + rclcpp::Duration::from_seconds(robot_pose_wait_timeout_);
  while (now() < deadline) {
    try {
      const auto trans = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
      const double age = (now() - rclcpp::Time(trans.header.stamp)).seconds();
      if (age > max_robot_pose_age_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "TF 位姿过旧 age=%.2fs", age);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      PoseStamped pose;
      pose.header.stamp = trans.header.stamp;
      pose.header.frame_id = "map";
      pose.pose.position.x = trans.transform.translation.x;
      pose.pose.position.y = trans.transform.translation.y;
      pose.pose.orientation = trans.transform.rotation;
      return pose;
    } catch (const tf2::LookupException & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "%s", e.what());
    } catch (const tf2::ConnectivityException & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "%s", e.what());
    } catch (const tf2::ExtrapolationException & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "%s", e.what());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return std::nullopt;
}

void CarAvoidancePointServer::execute(const std::shared_ptr<GoalHandle> goal_handle)
{
  std::lock_guard<std::mutex> execute_lock(execute_mutex_);
  RCLCPP_INFO(get_logger(), "开始寻找避让点...");
  startTaskSubscriptions();
  try {
    runTask(goal_handle);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Error raised in execute callback: %s", e.what());
  }
  if (goal_handle->is_active()) {
    goal_handle->abort(std::make_shared<FindCarAvoidancePoint::Result>());
  }
  stopTaskSubscriptions();
}

void CarAvoidancePointServer::runTask(const std::shared_ptr<GoalHandle> & goal_handle)
{
  auto empty_result = std::make_shared<FindCarAvoidancePoint::Result>();
  if (!waitForGlobalCostmap(2.0)) {
    RCLCPP_ERROR(get_logger(), "等待全局代价地图超时");
    goal_handle->abort(empty_result);
    return;
  }
  waitForSpecialTerrain(2.0);

  // 获取一份新鲜的机器人位姿，任务全程固定使用
  const auto robot_pose = getFreshRobotPose();
  if (!robot_pose) {
    RCLCPP_ERROR(get_logger(), "无法获取新鲜的机器人位姿，abort");
    goal_handle->abort(empty_result);
    return;
  }

  Task task;
  task.goal = goal_handle->get_goal();
  task.robot_pose = *robot_pose;
  RCLCPP_INFO(get_logger(), "polygons: %zu个通道", task.goal->polygons.size());
  publishAllPassagesMarker(task.goal->polygons);

  RCLCPP_INFO(get_logger(), "寻找当前通行区域...");
  const Vec2 robot(task.robot_pose.pose.position.x, task.robot_pose.pose.position.y);
  for (const auto & polygon : task.goal->polygons) {
    Ring current_polygon_vertices;
    for (const auto & point : polygon.points) {
      current_polygon_vertices.emplace_back(point.x, point.y);
    }
    if (pointInPolygon(robot, current_polygon_vertices)) {
      task.vertices = current_polygon_vertices;
      RCLCPP_INFO(get_logger(), "机器人在通道内，使用当前通道: [%s]", formatRing(task.vertices).c_str());
      break;
    }
  }
  if (task.vertices.empty()) {
    RCLCPP_ERROR(get_logger(), "未找到用于寻找停靠点的通道");
    goal_handle->abort(empty_result);
    return;
  }
  RCLCPP_INFO(get_logger(), "当前通行区域: [%s]", formatRing(task.vertices).c_str());

  std::vector<garage_road_model::RoadModelData> models;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    models = cached_road_models_;
  }
  std::string match_reason;
  auto matched = matchRoadModel(task.vertices, models, match_reason);
  const bool model_usable = matched && matched->valid &&
    matched->side_a.samples.size() >= 2 && matched->side_b.samples.size() >= 2;
  if (model_usable) {
    task.road_model = std::move(*matched);
  } else {
    if (matched && !matched->valid) {
      match_reason = matched->reason.empty() ? "模型无效" : matched->reason;
    } else if (matched) {
      match_reason = "侧边采样不足";
    } else if (match_reason.empty()) {
      match_reason = "未匹配到有效模型";
    }
    RCLCPP_ERROR(get_logger(), "没有可用曲线（%s）", match_reason.c_str());
    goal_handle->abort(empty_result);
    return;
  }

  auto search = selectAvoidanceFoot(task);
  if (!search || !normalAtFoot(*search) || !snapshotCostmap(*search)) {
    goal_handle->abort(empty_result);
    return;
  }

  if (const auto external = findExternalPoint(task, *search)) {
    auto result = std::make_shared<FindCarAvoidancePoint::Result>();
    result->pose = *external;
    goal_handle->succeed(result);
    if (external->pose.position.z >= 25.0) {
      RCLCPP_INFO(get_logger(), "优先使用通道外外部停靠点，保留上位机朝向，z=%.0f", external->pose.position.z);
    } else {
      RCLCPP_INFO(get_logger(), "优先使用通道内外部停靠点，朝向为沿侧边背对来车的切线，z=%.0f",
        external->pose.position.z);
    }
    return;
  }

  RCLCPP_INFO(get_logger(), "寻找停靠点...");
  const auto avoidance_point = findAvoidancePoint(task, *search);
  if (!avoidance_point) {
    RCLCPP_ERROR(get_logger(), "未找到避让点");
    goal_handle->abort(empty_result);
    return;
  }
  const auto & ap = avoidance_point->pose;
  RCLCPP_INFO(get_logger(), "成功找到避让点: x=%.3f, y=%.3f, z=%.3f, yaw=%.3f°",
    ap.position.x, ap.position.y, ap.position.z, radToDeg(tf2::getYaw(ap.orientation)));

  auto result = std::make_shared<FindCarAvoidancePoint::Result>();
  result->pose = *avoidance_point;
  goal_handle->succeed(result);
  const auto & rp = result->pose.pose;
  RCLCPP_INFO(get_logger(), "成功找到避让点*****: x=%.3f, y=%.3f, z=%.3f, yaw=%.3f°",
    rp.position.x, rp.position.y, rp.position.z, radToDeg(tf2::getYaw(rp.orientation)));
}

void CarAvoidancePointServer::publishDebugMarkers(
  const Vec2 & robot, const Vec2 & car, const Ring & side_line,
  const Ring * search_band, const PoseStamped * avoidance_point)
{
  using visualization_msgs::msg::Marker;
  const auto stamp = now();

  Marker car_marker;
  car_marker.header.frame_id = "map";
  car_marker.header.stamp = stamp;
  car_marker.id = 1;
  car_marker.type = Marker::CUBE;
  car_marker.action = Marker::ADD;
  car_marker.scale.x = 0.5;
  car_marker.scale.y = 0.5;
  car_marker.scale.z = 0.5;
  car_marker.color.r = 1.0;
  car_marker.color.a = 1.0;
  car_marker.pose.position.x = car.x();
  car_marker.pose.position.y = car.y();
  car_marker.pose.orientation.w = 1.0;
  marker_car_pose_publisher_->publish(car_marker);

  Marker robot_marker;
  robot_marker.header.frame_id = "map";
  robot_marker.header.stamp = stamp;
  robot_marker.id = 2;
  robot_marker.type = Marker::SPHERE;
  robot_marker.action = Marker::ADD;
  robot_marker.scale.x = 0.2;
  robot_marker.scale.y = 0.2;
  robot_marker.scale.z = 0.2;
  robot_marker.color.g = 1.0;
  robot_marker.color.a = 1.0;
  robot_marker.pose.position.x = robot.x();
  robot_marker.pose.position.y = robot.y();
  robot_marker.pose.orientation.w = 1.0;
  marker_robot_pose_publisher_->publish(robot_marker);

  Marker side_marker;
  side_marker.header.frame_id = "map";
  side_marker.header.stamp = stamp;
  side_marker.id = 3;
  side_marker.type = Marker::LINE_STRIP;
  side_marker.action = Marker::ADD;
  side_marker.scale.x = 0.2;
  side_marker.color.g = 1.0;
  side_marker.color.a = 1.0;
  for (const auto & point : side_line) {
    geometry_msgs::msg::Point marker_point;
    marker_point.x = point.x();
    marker_point.y = point.y();
    side_marker.points.push_back(marker_point);
  }
  if (side_marker.points.size() >= 2) {
    marker_avoidance_side_publisher_->publish(side_marker);
  }

  if (search_band) {
    Marker rect_marker;
    rect_marker.header.frame_id = "map";
    rect_marker.header.stamp = stamp;
    rect_marker.id = 4;
    rect_marker.type = Marker::LINE_LIST;
    rect_marker.action = Marker::ADD;
    rect_marker.scale.x = 0.1;
    rect_marker.color.r = 1.0;
    rect_marker.color.a = 1.0;
    const size_t size = search_band->size();
    for (size_t i = 0; i < size; ++i) {
      geometry_msgs::msg::Point start;
      start.x = (*search_band)[i].x();
      start.y = (*search_band)[i].y();
      geometry_msgs::msg::Point end;
      end.x = (*search_band)[(i + 1) % size].x();
      end.y = (*search_band)[(i + 1) % size].y();
      rect_marker.points.push_back(start);
      rect_marker.points.push_back(end);
    }
    marker_searching_rect_publisher_->publish(rect_marker);
  }

  if (avoidance_point) {
    Marker point_marker;
    point_marker.header.frame_id = "map";
    point_marker.header.stamp = stamp;
    point_marker.id = 5;
    point_marker.type = Marker::ARROW;
    point_marker.action = Marker::ADD;
    point_marker.scale.x = 0.5;
    point_marker.scale.y = 0.2;
    point_marker.scale.z = 0.4;
    point_marker.color.b = 1.0;
    point_marker.color.a = 1.0;
    point_marker.pose = avoidance_point->pose;
    marker_parking_point_publisher_->publish(point_marker);
  }
}

void CarAvoidancePointServer::publishAllPassagesMarker(const std::vector<geometry_msgs::msg::Polygon> & polygons)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = now();
  marker.id = 20;
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.03;
  marker.color.r = 0.0;
  marker.color.g = 0.5;
  marker.color.b = 1.0;
  marker.color.a = 0.8;
  if (polygons.empty()) {
    marker.action = visualization_msgs::msg::Marker::DELETE;
    marker_all_passages_publisher_->publish(marker);
    return;
  }
  for (const auto & polygon : polygons) {
    const auto & pts = polygon.points;
    if (pts.size() < 2) {
      continue;
    }
    for (size_t i = 0; i < pts.size(); ++i) {
      geometry_msgs::msg::Point p1;
      p1.x = pts[i].x;
      p1.y = pts[i].y;
      p1.z = pts[i].z;
      geometry_msgs::msg::Point p2;
      p2.x = pts[(i + 1) % pts.size()].x;
      p2.y = pts[(i + 1) % pts.size()].y;
      p2.z = pts[(i + 1) % pts.size()].z;
      marker.points.push_back(p1);
      marker.points.push_back(p2);
    }
  }
  marker_all_passages_publisher_->publish(marker);
}

Ring CarAvoidancePointServer::footprint() const
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  return footprint_vertices_;
}

}  // namespace find_free_space
