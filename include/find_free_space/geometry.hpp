#ifndef FIND_FREE_SPACE__GEOMETRY_HPP_
#define FIND_FREE_SPACE__GEOMETRY_HPP_

#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace find_free_space
{

using Vec2 = Eigen::Vector2d;
using Ring = std::vector<Vec2>;
using Edge = std::pair<Vec2, Vec2>;

double degToRad(double degrees);
double radToDeg(double radians);

// 点到线段（不是无限长直线）的最短距离
double distancePointToSegment(const Vec2 & point, const Vec2 & a, const Vec2 & b);

// 射线法判断点是否在多边形内部
bool pointInPolygon(const Vec2 & point, const Ring & vertices);

// 两角之差的绝对值，只做一次 ±2π 折返
double angleDiff(double a, double b);

// 两角相加并限制到 [-π, π]
double addAngles(double current_angle, double rotation_angle);

// 在 target_angle 与其反向之间，选与 direction 夹角更小的一个，输入输出都是角度制
double adjustAngle(const Vec2 & direction, double target_angle);

// 求点 c 在直线 ab 上的垂足，再沿法向往 c 的另一侧移动 distance
std::optional<Vec2> findIntersection(const Vec2 & a, const Vec2 & b, const Vec2 & c, double distance);

// 两条最长边中，到 robot 的直线距离更近的一条
std::optional<Edge> findNearestBoundary(const Vec2 & robot, const Ring & vertices);

// 四条边中最短的边长
double minEdgeLength(const Ring & vertices);

// 两条最短边中，中点沿 k 方向投影最大的一条
std::optional<Edge> forwardShortEdge(const Vec2 & robot, const Ring & vertices, double k);

// point 是否越过 edge 到多边形外侧
bool isPointBeyondEdge(const Vec2 & point, const Edge & edge, const Vec2 & polygon_center);

// 四边形顶点按 [左上, 右上, 右下, 左下] 排序
Ring sortQuadrilateralVertices(const Ring & points);

Ring convexHull(Ring points);

bool pointInConvex(const Vec2 & point, const Ring & polygon);

// 以 (center_x, center_y) 为中心、边长 resolution 的栅格是否与凸多边形有接触
bool cellIntersectsConvex(double center_x, double center_y, double resolution, const Ring & polygon);

// 局部 footprint 旋转 yaw 后平移到 (x, y)
Ring transformFootprint(const Ring & local, double x, double y, double yaw);

}  // namespace find_free_space

#endif  // FIND_FREE_SPACE__GEOMETRY_HPP_
