#include "find_free_space/car_avoidance_point_server.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "garage_road_model/polygon_processor.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace find_free_space
{

namespace
{

// 通道外的车辆中心离侧边超过该距离时，不再把它当成当前道路上的来车。
constexpr double kOutsideRoadLimit = 2.0;
// 点落在横截面附近才算属于这一段。下限覆盖采样误差，并随路宽放宽。
constexpr double kSectionSlack = 0.5;

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
  tf2::Quaternion quaternion;
  quaternion.setRPY(0.0, 0.0, yaw);
  return tf2::toMsg(quaternion);
}

struct SideAnchor
{
  enum class Status
  {
    Ok,
    Ambiguous,
    Miss
  };

  Status status{Status::Miss};
  double s{0.0};
  double distance{std::numeric_limits<double>::infinity()};
  double lateral{0.0};
  bool has_lateral{false};
  Vec2 foot{Vec2::Zero()};
  Vec2 outward{Vec2::Zero()};
  std::string reason;
};

bool pointNearSection(const Vec2 & point, const garage_road_model::CrossSection & section)
{
  const double limit = std::max(kSectionSlack, 0.25 * section.width);
  return distancePointToSegment(point, section.near_point, section.far_point) <= limit;
}

// 整条侧边投影。投影落在 U 形另一条臂上仍然有效；多个远隔投影，或投影位置对不上这段道路区域，才拒绝。
// 不用弧长窗口把另一条臂提前排除。
SideAnchor associateToSide(
  const garage_road_model::RoadModelQuery & query,
  garage_road_model::SideId side,
  const Vec2 & point,
  bool require_region)
{
  SideAnchor anchor;
  const auto projection = query.projectOnSide(side, point, 0.0, -1.0);
  if (projection.status == garage_road_model::QueryStatus::Ambiguous) {
    anchor.status = SideAnchor::Status::Ambiguous;
    anchor.reason = projection.reason.empty() ? "存在多个合理投影，无法确定所属区段" : projection.reason;
    return anchor;
  }
  if (projection.status != garage_road_model::QueryStatus::Ok) {
    anchor.reason = projection.reason.empty() ? "侧边无法投影" : projection.reason;
    return anchor;
  }

  const bool inside = garage_road_model::pointInsidePolygon(point, query.model().boundary, 0.05);
  if (inside) {
    const auto section = query.localCrossSection(side, projection.s);
    if (section.status == garage_road_model::QueryStatus::Ambiguous) {
      anchor.status = SideAnchor::Status::Ambiguous;
      anchor.reason = section.reason.empty() ? "投影位置无法对应道路区域" : section.reason;
      return anchor;
    }
    bool belongs = section.status == garage_road_model::QueryStatus::Ok && pointNearSection(point, section);
    if (!belongs && section.status != garage_road_model::QueryStatus::Ok) {
      const auto & samples = side == garage_road_model::SideId::A ?
        query.model().side_a.samples : query.model().side_b.samples;
      if (samples.size() >= 2) {
        const double begin = samples.front().s;
        const double end = samples.back().s;
        for (const double delta : {-0.2, 0.2, -0.4, 0.4}) {
          const double neighbor = std::clamp(projection.s + delta, begin, end);
          const auto neighbor_section = query.localCrossSection(side, neighbor);
          if (neighbor_section.status == garage_road_model::QueryStatus::Ok &&
            pointNearSection(point, neighbor_section))
          {
            belongs = true;
            break;
          }
        }
      }
    }
    if (!belongs) {
      anchor.status = SideAnchor::Status::Ambiguous;
      anchor.reason = section.status == garage_road_model::QueryStatus::Ok || section.reason.empty() ?
        "投影位置与道路横截面不符" : section.reason;
      return anchor;
    }
  } else if (require_region && projection.distance > kOutsideRoadLimit) {
    anchor.reason = "不在当前道路区域";
    return anchor;
  }

  anchor.status = SideAnchor::Status::Ok;
  anchor.s = projection.s;
  anchor.distance = projection.distance;
  anchor.foot = projection.point;
  const auto inward = query.inwardNormal(side, projection.s);
  if (inward.status == garage_road_model::QueryStatus::Ok) {
    anchor.outward = Vec2(-inward.direction.x(), -inward.direction.y());
    anchor.lateral = (point - anchor.foot).dot(anchor.outward);
    anchor.has_lateral = true;
  }
  return anchor;
}

std::optional<double> orientedYaw(
  const garage_road_model::RoadModelQuery & query,
  garage_road_model::SideId side,
  double s,
  double away_sign)
{
  const auto pose = query.evaluateSide(side, s);
  if (pose.status != garage_road_model::QueryStatus::Ok || pose.corner) {
    return std::nullopt;
  }
  double yaw = pose.yaw;
  if (away_sign < 0.0) {
    yaw = addAngles(yaw, M_PI);
  }
  return yaw;
}

const garage_road_model::SideBoundary & boundaryOf(
  const garage_road_model::RoadModelQuery & query, garage_road_model::SideId side)
{
  return side == garage_road_model::SideId::A ? query.model().side_a : query.model().side_b;
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

  if (!task.road_model) {
    RCLCPP_ERROR(get_logger(), "模型缺失：本次任务没有道路模型快照");
    return std::nullopt;
  }
  ctx.query = std::make_shared<garage_road_model::RoadModelQuery>(*task.road_model);
  ctx.robot = Vec2(task.robot_pose.pose.position.x, task.robot_pose.pose.position.y);
  ctx.robot_yaw = tf2::getYaw(task.robot_pose.pose.orientation);
  ctx.car = Vec2(task.goal->car_pose.pose.position.x, task.goal->car_pose.pose.position.y);

  const SideAnchor side_a = associateToSide(*ctx.query, garage_road_model::SideId::A, ctx.robot, true);
  const SideAnchor side_b = associateToSide(*ctx.query, garage_road_model::SideId::B, ctx.robot, true);
  const SideAnchor * chosen = nullptr;
  if (side_a.status == SideAnchor::Status::Ok &&
    (side_b.status != SideAnchor::Status::Ok || side_a.distance <= side_b.distance))
  {
    chosen = &side_a;
    ctx.side = garage_road_model::SideId::A;
  } else if (side_b.status == SideAnchor::Status::Ok) {
    chosen = &side_b;
    ctx.side = garage_road_model::SideId::B;
  }
  if (!chosen) {
    const bool ambiguous =
      side_a.status == SideAnchor::Status::Ambiguous || side_b.status == SideAnchor::Status::Ambiguous;
    if (ambiguous) {
      RCLCPP_ERROR(get_logger(), "存在多个合理投影，无法确定机器人所属区段：side_a=%s，side_b=%s",
        side_a.reason.c_str(), side_b.reason.c_str());
    } else {
      RCLCPP_ERROR(get_logger(), "侧边无法查询：side_a=%s，side_b=%s",
        side_a.reason.c_str(), side_b.reason.c_str());
    }
    return std::nullopt;
  }
  ctx.robot_s = chosen->s;
  RCLCPP_INFO(get_logger(), "近侧边: %s，机器人弧长 %.3f，投影距离 %.3f",
    ctx.side == garage_road_model::SideId::A ? "side_a" : "side_b", ctx.robot_s, chosen->distance);

  const SideAnchor car_anchor = associateToSide(*ctx.query, ctx.side, ctx.car, true);
  if (car_anchor.status != SideAnchor::Status::Ok) {
    if (car_anchor.status == SideAnchor::Status::Ambiguous) {
      RCLCPP_ERROR(get_logger(), "车辆投影不明确：存在多个合理投影，无法确定所属区段（%s）",
        car_anchor.reason.c_str());
    } else {
      RCLCPP_ERROR(get_logger(), "车辆无法关联到所选侧边：%s", car_anchor.reason.c_str());
    }
    return std::nullopt;
  }
  ctx.car_s = car_anchor.s;
  if (std::abs(ctx.car_s - ctx.robot_s) < 1e-3) {
    RCLCPP_ERROR(get_logger(), "车辆与机器人在所选侧边上的弧长重合，无法判断来车方向");
    return std::nullopt;
  }
  ctx.away_sign = ctx.car_s > ctx.robot_s ? -1.0 : 1.0;
  RCLCPP_INFO(get_logger(), "车辆弧长 %.3f，沿路搜索方向 %s",
    ctx.car_s, ctx.away_sign > 0.0 ? "弧长增大" : "弧长减小");

  const auto & samples = boundaryOf(*ctx.query, ctx.side).samples;
  ctx.side_line.reserve(samples.size());
  for (const auto & sample : samples) {
    ctx.side_line.emplace_back(sample.x, sample.y);
  }
  publishDebugMarkers(ctx.robot, ctx.car, ctx.side_line);
  return ctx;
}

std::vector<CarAvoidancePointServer::ExternalCandidate> CarAvoidancePointServer::filterExternalStopPoints(
  const SearchContext & ctx)
{
  geometry_msgs::msg::PoseArray::ConstSharedPtr stop_points;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stop_points = vehicle_avoidance_stop_points_;
  }
  if (!stop_points || stop_points->poses.empty() || !ctx.query) {
    return {};
  }

  std::vector<ExternalCandidate> results;
  for (const auto & pose : stop_points->poses) {
    const Vec2 point(pose.position.x, pose.position.y);
    if ((point - ctx.robot).norm() > external_point_max_distance_) {
      continue;
    }
    // 通道外的点不要求落在通道内。投影不明确就淘汰这一点，继续看其他点。
    const SideAnchor anchor = associateToSide(*ctx.query, ctx.side, point, false);
    if (anchor.status != SideAnchor::Status::Ok) {
      RCLCPP_INFO(get_logger(), "外部停车点 (%.2f, %.2f) %s，跳过",
        point.x(), point.y(),
        anchor.status == SideAnchor::Status::Ambiguous ? "投影不明确" : "无法关联到所选侧边");
      continue;
    }
    const double point_delta = anchor.s - ctx.robot_s;
    const double car_delta = ctx.car_s - ctx.robot_s;
    if (point_delta * car_delta >= -1e-4) {
      RCLCPP_INFO(get_logger(), "外部停车点 (%.2f, %.2f) 与车在沿路同侧，过滤掉", point.x(), point.y());
      continue;
    }
    ExternalCandidate candidate;
    candidate.pose.header.stamp = now();
    candidate.pose.header.frame_id = "map";
    candidate.pose.pose.position.x = point.x();
    candidate.pose.pose.position.y = point.y();
    candidate.pose.pose.position.z = 0.0;
    candidate.pose.pose.orientation = pose.orientation;
    candidate.along_s = anchor.s;
    candidate.side_distance = anchor.distance;
    results.push_back(std::move(candidate));
  }
  RCLCPP_INFO(get_logger(), "外部停车点过滤后剩余: %zu个", results.size());
  return results;
}

std::optional<CarAvoidancePointServer::PoseStamped> CarAvoidancePointServer::findAvoidancePoint(const Task & task)
{
  const auto prepared = prepareAvoidanceSearch(task);
  if (!prepared) {
    return std::nullopt;
  }
  const SearchContext & ctx = *prepared;

  const auto external_candidates = filterExternalStopPoints(ctx);
  if (!external_candidates.empty()) {
    RCLCPP_INFO(get_logger(), "尝试外部停车点，共%zu个", external_candidates.size());
    std::vector<ExternalCandidate> valid_external;
    for (auto candidate : external_candidates) {
      const Vec2 point(candidate.pose.pose.position.x, candidate.pose.pose.position.y);
      candidate.inside = pointInPolygon(point, task.vertices);
      if (candidate.inside) {
        const auto yaw = orientedYaw(*ctx.query, ctx.side, candidate.along_s, ctx.away_sign);
        if (!yaw) {
          RCLCPP_INFO(get_logger(), "外部停车点(%.2f, %.2f) 在通道内，但该处切向不明确，跳过",
            point.x(), point.y());
          continue;
        }
        candidate.pose.pose.orientation = yawToQuaternion(*yaw);
      } else {
        RCLCPP_INFO(get_logger(), "外部停车点(%.2f, %.2f) 在通道外，保留上位机朝向，将跳过服务检查",
          point.x(), point.y());
      }
      std::string reason;
      if (validateCandidate(task, ctx, candidate.pose, candidate.along_s, !candidate.inside, nullptr, reason)) {
        valid_external.push_back(std::move(candidate));
      } else {
        RCLCPP_INFO(get_logger(), "外部停车点(%.2f, %.2f) %s", point.x(), point.y(), reason.c_str());
      }
    }

    if (!valid_external.empty()) {
      const auto best = std::min_element(valid_external.begin(), valid_external.end(),
          [](const ExternalCandidate & lhs, const ExternalCandidate & rhs) {
            return lhs.side_distance < rhs.side_distance;
          });
      PoseStamped result = best->pose;
      const Vec2 best_point(result.pose.position.x, result.pose.position.y);
      if (best->inside) {
        result.pose.position.z = 20.0;
        RCLCPP_INFO(get_logger(),
          "外部停车点通过校验共%zu个，选择最贴近所选侧边: (%.2f, %.2f)，该点在通道内，使用局部切向，z = 20.0",
          valid_external.size(), best_point.x(), best_point.y());
      } else {
        result.pose.position.z = 30.0;
        RCLCPP_INFO(get_logger(),
          "外部停车点通过校验共%zu个，选择最贴近所选侧边: (%.2f, %.2f)，该点在通道外，方向保持不变，z = 30.0",
          valid_external.size(), best_point.x(), best_point.y());
        RCLCPP_INFO(get_logger(), "通道外外部停靠点已选用，map yaw=%.1f°",
          radToDeg(tf2::getYaw(result.pose.orientation)));
      }
      publishDebugMarkers(ctx.robot, ctx.car, ctx.side_line, nullptr, &result);
      return result;
    }
    RCLCPP_INFO(get_logger(), "有外部停车点，但是所有外部停车点在筛选判断以后均不满足，回退自搜索");
  } else {
    RCLCPP_INFO(get_logger(), "话题无有效外部停车点，使用自搜索");
  }

  const auto & side_boundary = boundaryOf(*ctx.query, ctx.side);
  const double side_begin = side_boundary.samples.front().s;
  const double side_end = side_boundary.samples.back().s;
  bool forward = false;
  std::optional<double> travel_yaw = orientedYaw(*ctx.query, ctx.side, ctx.robot_s, ctx.away_sign);
  if (!travel_yaw) {
    for (int step = 1; step <= 12; ++step) {
      const double probe = ctx.robot_s + ctx.away_sign * 0.05 * step;
      travel_yaw = orientedYaw(*ctx.query, ctx.side, probe, ctx.away_sign);
      if (travel_yaw) {
        break;
      }
    }
  }
  if (!travel_yaw) {
    RCLCPP_ERROR(get_logger(), "机器人处没有指向搜索方向的切线，结束自搜索");
    return std::nullopt;
  }
  forward = angleDiff(*travel_yaw, ctx.robot_yaw) < M_PI / 2.0;
  const double extra = forward ? search_radius_extra_dis_ : 0.0;
  if (forward) {
    RCLCPP_INFO(get_logger(), "在机器人前方避车, 额外增加%f米搜索距离", search_radius_extra_dis_);
  } else {
    RCLCPP_INFO(get_logger(), "在机器人后方避车");
  }
  const double near_distance = search_radius_min_ + extra;
  const double far_distance = search_radius_max_ + extra;
  const double s_near = ctx.robot_s + ctx.away_sign * near_distance;
  const double s_far = ctx.robot_s + ctx.away_sign * far_distance;
  if (s_near < side_begin || s_near > side_end || s_far < side_begin || s_far > side_end) {
    RCLCPP_ERROR(get_logger(), "搜索带越过所选侧边端头，结束自搜索");
    return std::nullopt;
  }

  const double lat_lo = std::min(outside_min_, outside_max_);
  const double lat_hi = std::max(outside_min_, outside_max_);
  const double s_lo = std::min(s_near, s_far);
  const double s_hi = std::max(s_near, s_far);
  RCLCPP_INFO(get_logger(),
    "沿路搜索带弧长 [%.3f, %.3f]，横向偏移 [%.2f, %.2f]（向外为正）",
    s_lo, s_hi, lat_lo, lat_hi);

  Ring inner_edge;
  Ring outer_edge;
  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  auto include_point = [&](const Vec2 & point) {
      min_x = std::min(min_x, point.x());
      max_x = std::max(max_x, point.x());
      min_y = std::min(min_y, point.y());
      max_y = std::max(max_y, point.y());
    };
  const double sample_step = std::max(0.2, ctx.map.resolution);
  for (double s = s_lo; s <= s_hi + 1e-6; s += sample_step) {
    const double station = std::min(s, s_hi);
    const auto inward = ctx.query->inwardNormal(ctx.side, station);
    const auto pose = ctx.query->evaluateSide(ctx.side, station);
    if (inward.status != garage_road_model::QueryStatus::Ok ||
      pose.status != garage_road_model::QueryStatus::Ok)
    {
      continue;
    }
    const Vec2 outward(-inward.direction.x(), -inward.direction.y());
    const Vec2 inner = pose.position + outward * lat_lo;
    const Vec2 outer = pose.position + outward * lat_hi;
    inner_edge.push_back(inner);
    outer_edge.push_back(outer);
    include_point(inner);
    include_point(outer);
  }
  if (inner_edge.empty()) {
    RCLCPP_ERROR(get_logger(), "搜索带上没有可用法线，结束自搜索");
    return std::nullopt;
  }
  Ring search_band = inner_edge;
  search_band.insert(search_band.end(), outer_edge.rbegin(), outer_edge.rend());
  publishDebugMarkers(ctx.robot, ctx.car, ctx.side_line, &search_band);

  struct CellCandidate
  {
    Vec2 point{Vec2::Zero()};
    double along_s{0.0};
    double along_distance{0.0};
    int order{0};
  };
  std::vector<CellCandidate> cells;
  int ambiguous_cells = 0;
  if (std::isfinite(min_x)) {
    const double pad = ctx.map.resolution;
    auto to_cell = [](double value, double origin, double resolution, int size) {
        return std::clamp(static_cast<int>(std::floor((value - origin) / resolution)), 0, size - 1);
      };
    const int x0 = to_cell(min_x - pad, ctx.map.origin_x, ctx.map.resolution, ctx.map.width);
    const int x1 = to_cell(max_x + pad, ctx.map.origin_x, ctx.map.resolution, ctx.map.width);
    const int y0 = to_cell(min_y - pad, ctx.map.origin_y, ctx.map.resolution, ctx.map.height);
    const int y1 = to_cell(max_y + pad, ctx.map.origin_y, ctx.map.resolution, ctx.map.height);
    int order = 0;
    for (int cell_y = y0; cell_y <= y1; ++cell_y) {
      for (int cell_x = x0; cell_x <= x1; ++cell_x) {
        const Vec2 center(
          ctx.map.origin_x + (cell_x + 0.5) * ctx.map.resolution,
          ctx.map.origin_y + (cell_y + 0.5) * ctx.map.resolution);
        const int scan_order = order++;
        const bool inside = garage_road_model::pointInsidePolygon(center, ctx.query->model().boundary, 0.05);
        const SideAnchor anchor = associateToSide(*ctx.query, ctx.side, center, inside);
        if (anchor.status == SideAnchor::Status::Ambiguous) {
          ++ambiguous_cells;
          continue;
        }
        if (anchor.status != SideAnchor::Status::Ok || !anchor.has_lateral) {
          continue;
        }
        if (anchor.s < s_lo || anchor.s > s_hi || anchor.lateral < lat_lo || anchor.lateral > lat_hi) {
          continue;
        }
        CellCandidate cell;
        cell.point = center;
        cell.along_s = anchor.s;
        cell.along_distance = ctx.away_sign * (anchor.s - ctx.robot_s);
        cell.order = scan_order;
        cells.push_back(cell);
      }
    }
  }

  // 排序规则调整：按沿路距离由近到远，沿路距离相同再保持栅格扫描顺序。
  // 弯道里隔墙直线距离更近的点，不会因为欧氏距离更小而被优先选中。
  std::stable_sort(cells.begin(), cells.end(), [](const CellCandidate & lhs, const CellCandidate & rhs) {
      if (std::abs(lhs.along_distance - rhs.along_distance) > 1e-6) {
        return lhs.along_distance < rhs.along_distance;
      }
      return lhs.order < rhs.order;
    });
  RCLCPP_INFO(get_logger(),
    "自搜索候选 %zu 个，按沿路距离由近到远检查，投影不明确跳过 %d 个",
    cells.size(), ambiguous_cells);

  if (cells.empty()) {
    const double mid_s = 0.5 * (s_near + s_far);
    const double mid_lateral = 0.5 * (outside_min_ + outside_max_);
    const auto inward = ctx.query->inwardNormal(ctx.side, mid_s);
    const auto pose = ctx.query->evaluateSide(ctx.side, mid_s);
    if (inward.status != garage_road_model::QueryStatus::Ok ||
      pose.status != garage_road_model::QueryStatus::Ok)
    {
      RCLCPP_INFO(get_logger(), "搜索带内没有栅格中心，且中心处没有可用法线");
      return std::nullopt;
    }
    const Vec2 outward(-inward.direction.x(), -inward.direction.y());
    CellCandidate cell;
    cell.point = pose.position + outward * mid_lateral;
    cell.along_s = mid_s;
    cell.along_distance = ctx.away_sign * (mid_s - ctx.robot_s);
    cells.push_back(cell);
    RCLCPP_INFO(get_logger(), "搜索带内没有栅格中心，使用搜索带中心(%.3f, %.3f)作为候选点",
      cell.point.x(), cell.point.y());
  }

  std::map<std::string, int> rejected;
  auto log_rejected = [&]() {
      for (const auto & item : rejected) {
        RCLCPP_INFO(get_logger(), "候选栅格不通过：%s %d个", item.first.c_str(), item.second);
      }
    };
  for (const auto & cell : cells) {
    const auto yaw = orientedYaw(*ctx.query, ctx.side, cell.along_s, ctx.away_sign);
    if (!yaw) {
      ++rejected["该处切向不明确"];
      continue;
    }
    PoseStamped candidate;
    candidate.header.stamp = now();
    candidate.header.frame_id = "map";
    candidate.pose.position.x = cell.point.x();
    candidate.pose.position.y = cell.point.y();
    candidate.pose.orientation = yawToQuaternion(*yaw);
    std::string reason;
    // 各候选朝向不同，不能共用一份 footprint 栅格模板。
    if (validateCandidate(task, ctx, candidate, cell.along_s, false, nullptr, reason)) {
      log_rejected();
      candidate.pose.position.z = 10.0;
      publishDebugMarkers(ctx.robot, ctx.car, ctx.side_line, &search_band, &candidate);
      return candidate;
    }
    ++rejected[reason];
  }

  log_rejected();
  RCLCPP_INFO(get_logger(), "当前搜索区域没有找到避让点");
  return std::nullopt;
}

}  // namespace find_free_space
