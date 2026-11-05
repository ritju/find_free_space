#include "find_free_space/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace find_free_space
{

namespace
{

double orient(const Vec2 & a, const Vec2 & b, const Vec2 & c)
{
  return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
}

bool segmentsIntersect(const Vec2 & a, const Vec2 & b, const Vec2 & c, const Vec2 & d)
{
  const double o1 = orient(a, b, c);
  const double o2 = orient(a, b, d);
  const double o3 = orient(c, d, a);
  const double o4 = orient(c, d, b);
  return o1 * o2 < 0.0 && o3 * o4 < 0.0;
}

std::vector<Edge> polygonEdges(const Ring & vertices)
{
  return {
    {vertices.at(0), vertices.at(1)},
    {vertices.at(1), vertices.at(2)},
    {vertices.at(2), vertices.at(3)},
    {vertices.at(3), vertices.at(0)}};
}

}  // namespace

double degToRad(double degrees)
{
  return degrees * (M_PI / 180.0);
}

double radToDeg(double radians)
{
  return radians * (180.0 / M_PI);
}

double distancePointToSegment(const Vec2 & point, const Vec2 & a, const Vec2 & b)
{
  const double dx = b.x() - a.x();
  const double dy = b.y() - a.y();
  const double seg_len_sq = dx * dx + dy * dy;
  if (seg_len_sq == 0.0) {
    return (point - a).norm();
  }
  double t = ((point.x() - a.x()) * dx + (point.y() - a.y()) * dy) / seg_len_sq;
  t = std::max(0.0, std::min(1.0, t));
  return (point - Vec2(a.x() + t * dx, a.y() + t * dy)).norm();
}

bool pointInPolygon(const Vec2 & point, const Ring & vertices)
{
  bool inside = false;
  if (vertices.empty()) {
    return inside;
  }
  size_t j = vertices.size() - 1;
  for (size_t i = 0; i < vertices.size(); ++i) {
    const double xi = vertices[i].x();
    const double yi = vertices[i].y();
    const double xj = vertices[j].x();
    const double yj = vertices[j].y();
    const bool intersect = ((yi > point.y()) != (yj > point.y())) &&
      (point.x() < (xj - xi) * (point.y() - yi) / (yj - yi) + xi);
    if (intersect) {
      inside = !inside;
    }
    j = i;
  }
  return inside;
}

double angleDiff(double a, double b)
{
  double error = a - b;
  if (error < -M_PI) {
    error = error + 2.0 * M_PI;
  } else if (error >= M_PI) {
    error = error - 2.0 * M_PI;
  }
  return std::abs(error);
}

double addAngles(double current_angle, double rotation_angle)
{
  double new_angle = current_angle + rotation_angle;
  while (new_angle > M_PI) {
    new_angle -= 2.0 * M_PI;
  }
  while (new_angle < -M_PI) {
    new_angle += 2.0 * M_PI;
  }
  return new_angle;
}

double adjustAngle(const Vec2 & direction, double target_angle)
{
  const double target_angle_2 = target_angle > 0.0 ? target_angle - 180.0 : 180.0 + target_angle;
  const Vec2 v_target(std::cos(degToRad(target_angle)), std::sin(degToRad(target_angle)));
  const Vec2 v_target_2(std::cos(degToRad(target_angle_2)), std::sin(degToRad(target_angle_2)));
  if (direction.norm() == 0.0 || v_target.norm() == 0.0 || v_target_2.norm() == 0.0) {
    return target_angle;
  }
  const double cos_theta_1 = std::clamp(
    direction.dot(v_target) / (direction.norm() * v_target.norm()), -1.0, 1.0);
  const double cos_theta_2 = std::clamp(
    direction.dot(v_target_2) / (direction.norm() * v_target_2.norm()), -1.0, 1.0);
  return std::abs(std::acos(cos_theta_1)) < std::abs(std::acos(cos_theta_2)) ? target_angle : target_angle_2;
}

std::optional<Vec2> findIntersection(const Vec2 & a, const Vec2 & b, const Vec2 & c, double distance)
{
  const Vec2 ab = b - a;
  const double length_ab = ab.norm();
  if (length_ab == 0.0) {
    return std::nullopt;
  }
  const Vec2 unit_ab = ab / length_ab;
  const Vec2 normal(-unit_ab.y(), unit_ab.x());
  const Vec2 ac = c - a;
  const double projection = ac.dot(unit_ab);
  const Vec2 foot = a + projection * unit_ab;
  const double normal_projection = ac.dot(normal);
  const double direction = normal_projection > 0.0 ? -1.0 : 1.0;
  return Vec2(foot + direction * normal * distance);
}

std::optional<Edge> findNearestBoundary(const Vec2 & robot, const Ring & vertices)
{
  auto edges = polygonEdges(vertices);
  std::stable_sort(edges.begin(), edges.end(), [](const Edge & lhs, const Edge & rhs) {
      return (lhs.second - lhs.first).norm() > (rhs.second - rhs.first).norm();
    });

  double min_distance = std::numeric_limits<double>::infinity();
  std::optional<Edge> nearest;
  for (size_t i = 0; i < 2; ++i) {
    const Vec2 & p1 = edges[i].first;
    const Vec2 & p2 = edges[i].second;
    const double a = p1.y() - p2.y();
    const double b = p2.x() - p1.x();
    const double c = p1.x() * p2.y() - p2.x() * p1.y();
    const double distance = std::abs(a * robot.x() + b * robot.y() + c) / std::sqrt(a * a + b * b);
    if (distance < min_distance) {
      min_distance = distance;
      nearest = edges[i];
    }
  }
  return nearest;
}

double minEdgeLength(const Ring & vertices)
{
  double distance = std::numeric_limits<double>::infinity();
  for (const auto & edge : polygonEdges(vertices)) {
    distance = std::min(distance, (edge.second - edge.first).norm());
  }
  return distance;
}

std::optional<Edge> forwardShortEdge(const Vec2 & robot, const Ring & vertices, double k)
{
  auto edges = polygonEdges(vertices);
  std::stable_sort(edges.begin(), edges.end(), [](const Edge & lhs, const Edge & rhs) {
      return (lhs.second - lhs.first).norm() < (rhs.second - rhs.first).norm();
    });
  const Vec2 dir_vec(std::cos(k), std::sin(k));
  std::optional<Edge> best_edge;
  double best_proj = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < 2; ++i) {
    const Vec2 mid = (edges[i].first + edges[i].second) / 2.0;
    const double proj = (mid - robot).dot(dir_vec);
    if (proj > best_proj) {
      best_proj = proj;
      best_edge = edges[i];
    }
  }
  return best_edge;
}

bool isPointBeyondEdge(const Vec2 & point, const Edge & edge, const Vec2 & polygon_center)
{
  const Vec2 edge_vec = edge.second - edge.first;
  const Vec2 normal1(-edge_vec.y(), edge_vec.x());
  const Vec2 inward_normal = (polygon_center - edge.first).dot(normal1) >= 0.0 ? normal1 : Vec2(-normal1);
  return (point - edge.first).dot(inward_normal) < 0.0;
}

Ring sortQuadrilateralVertices(const Ring & points)
{
  Vec2 center = Vec2::Zero();
  for (const auto & point : points) {
    center += point;
  }
  center /= static_cast<double>(points.size());

  std::vector<size_t> by_angle(points.size());
  std::iota(by_angle.begin(), by_angle.end(), 0);
  std::stable_sort(by_angle.begin(), by_angle.end(), [&](size_t lhs, size_t rhs) {
      return std::atan2(points[lhs].y() - center.y(), points[lhs].x() - center.x()) <
             std::atan2(points[rhs].y() - center.y(), points[rhs].x() - center.x());
    });
  Ring sorted;
  for (size_t index : by_angle) {
    sorted.push_back(points[index]);
  }

  std::vector<size_t> by_y(sorted.size());
  std::iota(by_y.begin(), by_y.end(), 0);
  std::stable_sort(by_y.begin(), by_y.end(), [&](size_t lhs, size_t rhs) {
      return sorted[lhs].y() < sorted[rhs].y();
    });
  Vec2 top_0 = sorted[by_y[0]];
  Vec2 top_1 = sorted[by_y[1]];
  if (top_0.x() > top_1.x()) {
    std::swap(top_0, top_1);
  }
  Vec2 bottom_0 = sorted[by_y[2]];
  Vec2 bottom_1 = sorted[by_y[3]];
  if (bottom_0.x() < bottom_1.x()) {
    std::swap(bottom_0, bottom_1);
  }
  return {top_0, top_1, bottom_0, bottom_1};
}

Ring convexHull(Ring points)
{
  std::sort(points.begin(), points.end(), [](const Vec2 & a, const Vec2 & b) {
      return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
    });
  points.erase(
    std::unique(points.begin(), points.end(), [](const Vec2 & a, const Vec2 & b) {
      return (a - b).norm() < 1e-9;
    }),
    points.end());
  if (points.size() < 3) {
    return points;
  }
  Ring lower;
  Ring upper;
  for (const auto & point : points) {
    while (lower.size() >= 2 && orient(lower[lower.size() - 2], lower.back(), point) <= 0.0) {
      lower.pop_back();
    }
    lower.push_back(point);
  }
  for (auto it = points.rbegin(); it != points.rend(); ++it) {
    while (upper.size() >= 2 && orient(upper[upper.size() - 2], upper.back(), *it) <= 0.0) {
      upper.pop_back();
    }
    upper.push_back(*it);
  }
  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

bool pointInConvex(const Vec2 & point, const Ring & polygon)
{
  if (polygon.size() < 3) {
    return false;
  }
  bool positive = false;
  bool negative = false;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const double cross = orient(polygon[i], polygon[(i + 1) % polygon.size()], point);
    positive = positive || cross > 1e-8;
    negative = negative || cross < -1e-8;
    if (positive && negative) {
      return false;
    }
  }
  return true;
}

bool cellIntersectsConvex(double center_x, double center_y, double resolution, const Ring & polygon)
{
  const double half = 0.5 * resolution;
  const Vec2 cell[4] = {
    {center_x - half, center_y - half},
    {center_x + half, center_y - half},
    {center_x + half, center_y + half},
    {center_x - half, center_y + half}};
  for (const auto & corner : cell) {
    if (pointInConvex(corner, polygon)) {
      return true;
    }
  }
  for (const auto & point : polygon) {
    if (std::abs(point.x() - center_x) <= half && std::abs(point.y() - center_y) <= half) {
      return true;
    }
  }
  for (size_t i = 0; i < polygon.size(); ++i) {
    const auto & a = polygon[i];
    const auto & b = polygon[(i + 1) % polygon.size()];
    for (int j = 0; j < 4; ++j) {
      if (segmentsIntersect(a, b, cell[j], cell[(j + 1) % 4])) {
        return true;
      }
    }
  }
  return false;
}

Ring transformFootprint(const Ring & local, double x, double y, double yaw)
{
  const double cos_yaw = std::cos(yaw);
  const double sin_yaw = std::sin(yaw);
  Ring world;
  world.reserve(local.size());
  for (const auto & vertex : local) {
    world.emplace_back(
      x + vertex.x() * cos_yaw - vertex.y() * sin_yaw,
      y + vertex.x() * sin_yaw + vertex.y() * cos_yaw);
  }
  return world;
}

}  // namespace find_free_space
