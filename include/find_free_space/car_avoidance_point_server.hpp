#ifndef FIND_FREE_SPACE__CAR_AVOIDANCE_POINT_SERVER_HPP_
#define FIND_FREE_SPACE__CAR_AVOIDANCE_POINT_SERVER_HPP_

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "capella_ros_msg/action/find_car_avoidance_point.hpp"
#include "capella_ros_msg/srv/is_car_passable.hpp"
#include "find_free_space/geometry.hpp"
#include "garage_utils_msgs/msg/polygons.hpp"
#include "geometry_msgs/msg/polygon_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/msg/costmap.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker.hpp"

namespace find_free_space
{

class CarAvoidancePointServer : public rclcpp::Node
{
public:
  using FindCarAvoidancePoint = capella_ros_msg::action::FindCarAvoidancePoint;
  using GoalHandle = rclcpp_action::ServerGoalHandle<FindCarAvoidancePoint>;
  using IsCarPassable = capella_ros_msg::srv::IsCarPassable;
  using PoseStamped = geometry_msgs::msg::PoseStamped;

  CarAvoidancePointServer();

private:
  struct MapView
  {
    const uint8_t * data{nullptr};
    double origin_x{0.0};
    double origin_y{0.0};
    double resolution{0.0};
    int width{0};
    int height{0};

    uint8_t at(int x, int y) const
    {
      return data[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
    }
  };

  // 一次 action 请求内固定不变的输入
  struct Task
  {
    std::shared_ptr<const FindCarAvoidancePoint::Goal> goal;
    PoseStamped robot_pose;
    Ring vertices;
  };

  struct SearchContext
  {
    nav2_msgs::msg::Costmap::ConstSharedPtr costmap;
    MapView map;
    Vec2 robot;
    double robot_yaw{0.0};
    Vec2 car;
    Edge nearest_boundary;
    double k{0.0};
    double k_diff2{0.0};
  };

  // 中心落在栅格中心时，footprint 碰到的栅格相对中心栅格的偏移
  struct FootprintKernel
  {
    std::vector<std::pair<int, int>> offsets;
  };

  // car_avoidance_point_server.cpp
  void initParams();
  void footprintCallback(const geometry_msgs::msg::PolygonStamped::ConstSharedPtr msg);
  void globalCostmapCallback(const nav2_msgs::msg::Costmap::ConstSharedPtr msg);
  void specialTerrainCallback(const garage_utils_msgs::msg::Polygons::ConstSharedPtr msg);
  void vehicleStopPointsCallback(const geometry_msgs::msg::PoseArray::ConstSharedPtr msg);
  void startTaskSubscriptions();
  void stopTaskSubscriptions();
  bool waitForGlobalCostmap(double timeout_sec);
  void waitForSpecialTerrain(double timeout_sec);
  std::optional<PoseStamped> getFreshRobotPose();
  void execute(const std::shared_ptr<GoalHandle> goal_handle);
  void runTask(const std::shared_ptr<GoalHandle> & goal_handle);
  void publishDebugMarkers(
    const Vec2 & robot, const Vec2 & car, const Edge & nearest_boundary,
    const Ring * find_vertices = nullptr, const PoseStamped * avoidance_point = nullptr);
  void publishAllPassagesMarker(const std::vector<geometry_msgs::msg::Polygon> & polygons);
  Ring footprint() const;

  // avoidance_search.cpp
  std::optional<SearchContext> prepareAvoidanceSearch(const Task & task);
  std::optional<PoseStamped> findAvoidancePoint(const Task & task);
  std::vector<PoseStamped> filterExternalStopPoints(const SearchContext & ctx);
  std::vector<Vec2> searchCellCenters(const MapView & map, const Ring & find_vertices) const;

  // candidate_checks.cpp
  bool validateCandidate(
    const Task & task, const SearchContext & ctx, const PoseStamped & candidate,
    bool skip_service_check, const FootprintKernel * kernel, std::string & reason);
  bool checkPointIsFree(const MapView & map, int center_x, int center_y, int radius) const;
  bool isPointInSpecialTerrain(double x, double y) const;
  bool isFootprintInSpecialTerrain(const Ring & local_footprint, double x, double y, double yaw) const;
  bool polygonCellsFree(const MapView & map, const Ring & polygon) const;
  FootprintKernel buildFootprintKernel(const MapView & map, const Ring & local_footprint, double yaw) const;
  bool kernelCellsFree(const MapView & map, const FootprintKernel & kernel, int cell_x, int cell_y) const;
  bool checkPathIsFree(const MapView & map, int current_x, int current_y, int target_x, int target_y) const;
  bool checkFootprintSweep(
    const MapView & map, const Ring & local_footprint, const Vec2 & robot, const Vec2 & target,
    double target_yaw) const;
  bool checkAvoidance(const IsCarPassable::Request::SharedPtr request);

  // 参数
  std::string topic_name_global_costmap_;
  std::string topic_name_footprint_;
  double search_radius_min_{3.0};
  double search_radius_max_{4.0};
  double search_radius_extra_dis_{2.5};
  double outside_min_{0.0};
  double outside_max_{0.5};
  double check_service_max_time_{0.5};
  bool show_global_costmap_raw_cv2_{false};
  bool show_global_costmap_raw_colored_cv2_{false};
  double external_point_max_distance_{8.0};
  double point_free_check_radius_{0.2};
  double stop_in_place_min_dist_{2.0};
  double max_robot_pose_age_{1.0};
  double robot_pose_wait_timeout_{1.5};
  bool strict_check_mode_{false};

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_car_pose_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_robot_pose_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_avoidance_side_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_searching_rect_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_parking_point_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_special_terrain_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_all_passages_publisher_;

  rclcpp::CallbackGroup::SharedPtr action_cb_group_;
  rclcpp::CallbackGroup::SharedPtr costmap_cb_group_;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
  rclcpp::CallbackGroup::SharedPtr footprint_cb_group_;

  rclcpp::Subscription<geometry_msgs::msg::PolygonStamped>::SharedPtr footprint_sub_;
  rclcpp::Subscription<nav2_msgs::msg::Costmap>::SharedPtr global_costmap_sub_;
  rclcpp::Subscription<garage_utils_msgs::msg::Polygons>::SharedPtr special_terrain_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr vehicle_stop_points_sub_;
  rclcpp_action::Server<FindCarAvoidancePoint>::SharedPtr action_server_;
  rclcpp::Client<IsCarPassable>::SharedPtr check_avoidance_service_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // 订阅回调与 action 线程共享的数据
  mutable std::mutex data_mutex_;
  Ring footprint_vertices_;
  nav2_msgs::msg::Costmap::ConstSharedPtr global_costmap_;
  std::shared_ptr<const std::vector<geometry_msgs::msg::Polygon>> special_terrain_polygons_;
  geometry_msgs::msg::PoseArray::ConstSharedPtr vehicle_avoidance_stop_points_;

  std::mutex execute_mutex_;
};

}  // namespace find_free_space

#endif  // FIND_FREE_SPACE__CAR_AVOIDANCE_POINT_SERVER_HPP_
