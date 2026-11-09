#include "find_free_space/visualization.hpp"

namespace find_free_space
{
namespace
{

visualization_msgs::msg::Marker makeMarker(
  const builtin_interfaces::msg::Time & stamp,
  int id,
  int type,
  float scale_x,
  float scale_y,
  float scale_z,
  float r,
  float g,
  float b,
  float a)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = stamp;
  marker.id = id;
  marker.type = type;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = scale_x;
  marker.scale.y = scale_y;
  marker.scale.z = scale_z;
  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;
  marker.color.a = a;
  marker.pose.orientation.w = 1.0;
  return marker;
}

}  // namespace

DebugMarkers makeDebugMarkers(
  const builtin_interfaces::msg::Time & stamp,
  const Vec2 & robot,
  const Vec2 & car,
  const Ring & side_line,
  const Ring * search_band,
  const geometry_msgs::msg::PoseStamped * avoidance_point)
{
  using visualization_msgs::msg::Marker;
  DebugMarkers markers;
  markers.car = makeMarker(stamp, 1, Marker::CUBE, 0.5F, 0.5F, 0.5F, 1.F, 0.F, 0.F, 1.F);
  markers.car.pose.position.x = car.x();
  markers.car.pose.position.y = car.y();

  markers.robot = makeMarker(stamp, 2, Marker::SPHERE, 0.2F, 0.2F, 0.2F, 0.F, 1.F, 0.F, 1.F);
  markers.robot.pose.position.x = robot.x();
  markers.robot.pose.position.y = robot.y();

  if (side_line.size() >= 2) {
    auto side = makeMarker(stamp, 3, Marker::LINE_STRIP, 0.2F, 0.2F, 0.2F, 0.F, 1.F, 0.F, 1.F);
    for (const auto & point : side_line) {
      geometry_msgs::msg::Point marker_point;
      marker_point.x = point.x();
      marker_point.y = point.y();
      side.points.push_back(marker_point);
    }
    markers.side = std::move(side);
  }

  if (search_band != nullptr) {
    auto band = makeMarker(stamp, 4, Marker::LINE_LIST, 0.1F, 0.1F, 0.1F, 1.F, 0.F, 0.F, 1.F);
    const size_t size = search_band->size();
    for (size_t i = 0; i < size; ++i) {
      geometry_msgs::msg::Point start;
      start.x = (*search_band)[i].x();
      start.y = (*search_band)[i].y();
      geometry_msgs::msg::Point end;
      end.x = (*search_band)[(i + 1) % size].x();
      end.y = (*search_band)[(i + 1) % size].y();
      band.points.push_back(start);
      band.points.push_back(end);
    }
    markers.search_band = std::move(band);
  }

  if (avoidance_point != nullptr) {
    auto point = makeMarker(stamp, 5, Marker::ARROW, 0.5F, 0.2F, 0.4F, 0.F, 0.F, 1.F, 1.F);
    point.pose = avoidance_point->pose;
    markers.parking_point = std::move(point);
  }
  return markers;
}

visualization_msgs::msg::Marker makePolygonOutline(
  const builtin_interfaces::msg::Time & stamp,
  int id,
  float scale,
  float r,
  float g,
  float b,
  float a,
  const std::vector<geometry_msgs::msg::Polygon> & polygons)
{
  auto marker = makeMarker(
    stamp, id, visualization_msgs::msg::Marker::LINE_LIST, scale, scale, scale, r, g, b, a);
  if (polygons.empty()) {
    marker.action = visualization_msgs::msg::Marker::DELETE;
    return marker;
  }
  for (const auto & polygon : polygons) {
    const auto & points = polygon.points;
    if (points.size() < 2) {
      continue;
    }
    for (size_t i = 0; i < points.size(); ++i) {
      geometry_msgs::msg::Point start;
      start.x = points[i].x;
      start.y = points[i].y;
      start.z = points[i].z;
      geometry_msgs::msg::Point end;
      end.x = points[(i + 1) % points.size()].x;
      end.y = points[(i + 1) % points.size()].y;
      end.z = points[(i + 1) % points.size()].z;
      marker.points.push_back(start);
      marker.points.push_back(end);
    }
  }
  return marker;
}

}  // namespace find_free_space
