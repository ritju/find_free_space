#ifndef FIND_FREE_SPACE__VISUALIZATION_HPP_
#define FIND_FREE_SPACE__VISUALIZATION_HPP_

#include <optional>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "find_free_space/geometry.hpp"
#include "geometry_msgs/msg/polygon.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace find_free_space
{

struct DebugMarkers
{
  visualization_msgs::msg::Marker car;
  visualization_msgs::msg::Marker robot;
  std::optional<visualization_msgs::msg::Marker> side;
  std::optional<visualization_msgs::msg::Marker> search_band;
  std::optional<visualization_msgs::msg::Marker> parking_point;
};

// 车、机器人、避车长边、搜索框和停车点。搜索框和停车点没有时不生成。
DebugMarkers makeDebugMarkers(
  const builtin_interfaces::msg::Time & stamp,
  const Vec2 & robot,
  const Vec2 & car,
  const Ring & side_line,
  const Ring * search_band = nullptr,
  const geometry_msgs::msg::PoseStamped * avoidance_point = nullptr);

// 把一组多边形画成轮廓。没有多边形时返回删除标记。
visualization_msgs::msg::Marker makePolygonOutline(
  const builtin_interfaces::msg::Time & stamp,
  int id,
  float scale,
  float r,
  float g,
  float b,
  float a,
  const std::vector<geometry_msgs::msg::Polygon> & polygons);

}  // namespace find_free_space

#endif  // FIND_FREE_SPACE__VISUALIZATION_HPP_
