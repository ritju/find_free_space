import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import math

vertices = [(1, 0), (2, 2), (1, 1), (2, 1)]
vertices = [(8.9, 1.5), (13.9, 1.6), (13.9, 3.6), (8.9, 3.5)]
robot_pose = [(10, 0)]

def generate_all_serach_points(vertices, interval):
        # 顶点排序处理[6](@ref)
        A = np.array(vertices[0])
        B = np.array(vertices[1])
        D = np.array(vertices[3])
        
        # 计算基底向量[7](@ref)
        u = B - A  # AB边向量
        v = D - A  # AD边向量
        
        # 计算分割步数
        u_length = np.linalg.norm(u)
        v_length = np.linalg.norm(v)
        steps_u = max(1, int(u_length / interval))
        steps_v = max(1, int(v_length / interval))
        
        # 生成网格点[5](@ref)
        grid_points = defaultdict(list)
        for i in range(steps_u + 1):
            for j in range(steps_v + 1):
                point = A + u*(i/steps_u) + v*(j/steps_v)
                grid_points[(i, j)] = tuple(np.round(point, 6))
        
        # 生成小平行四边形顶点
        # cells = []
        # for i in range(steps_u):
        #     for j in range(steps_v):
        #         cell = [
        #             grid_points[(i, j)],
        #             grid_points[(i+1, j)], 
        #             grid_points[(i+1, j+1)],
        #             grid_points[(i, j+1)]
        #         ]
        #         cells.append(cell)
        
        generate_points = list(grid_points.values())
        # print(cells)
        # 去重
        # unique_points = list({tuple(p) for cell in grid_points for p in cell})
        generate_points = sorted(generate_points, key=distance_sq)
        return generate_points

def show_point(points, plot=True):
        # 控制台输出部分
        print(f"共 {len(points)} 个顶点坐标：")
        for i, (x, y) in enumerate(points):
            print(f"顶点{i+1}: ({x:.3f}, {y:.3f})")

        if plot:
            plt.figure(figsize=(8,6))
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            plt.scatter(xs, ys, c='red', s=100, zorder=3)
            
            for i, (x, y) in enumerate(points):
                plt.text(x, y, f' {i+1}\n({x:.2f},{y:.2f})', 
                        fontsize=9, ha='left', va='bottom')
            
            plt.title(f"顶点坐标可视化（共{len(points)}个点）")
            plt.grid(True)
            plt.xlabel('X轴')
            plt.ylabel('Y轴')
            plt.show()

def distance_sq(point):
        dx = point[0] - robot_pose[0][0]
        dy = point[1] - robot_pose[0][1]
        return dx**2 + dy**2

def generate_point_orientation(robot_pose, points, vertices):
      pass
      # 根据vertices中的四个顶点确定长度较长的两条边与x轴的夹角，根据robot_pose :[(x, y)], points为[（x0， y0）,(x1, yz), ...] 顶点列表，遍历points中的点，计算robot_pose指向每个point的方向与x轴的夹角α, 比较α与距离point距离近的长边的方向，调整α与该长边平行，且该α为调整的夹角最小的角度

def calculate_long_edges(vertices):
    """计算四边形最长的两条边及其与X轴夹角"""
    edges = []
    # 生成所有边（四边形顶点顺序需连续）
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i+1)%4]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        # angle = math.degrees(math.atan2(dy, dx)) % 180  # 统一为0-180°
        angle = math.degrees(math.atan2(dy, dx))
        edges.append((length, angle, (p1,p2)))
    
    # 按长度排序取前二
    sorted_edges = sorted(edges, key=lambda x: -x[0])
    return sorted_edges[0], sorted_edges[1]

def distance_point_to_line(point, line):
    """计算点到线段的最近距离"""
    (x0, y0), (x1, y1) = line
    px, py = point
    line_length = math.hypot(x1-x0, y1-y0)
    if line_length == 0:
        return math.hypot(px-x0, py-y0)
    
    t = ((px-x0)*(x1-x0) + (py-y0)*(y1-y0)) / line_length**2
    t = max(0, min(1, t))
    proj_x = x0 + t*(x1-x0)
    proj_y = y0 + t*(y1-y0)
    return math.hypot(px-proj_x, py-proj_y)

def adjust_angle(direction_vec, target_angle):

    if (target_angle > 0):
         target_angle_2 = target_angle - 180
    else:
         target_angle_2 = 180 + target_angle

    v_target_angle = (math.cos(target_angle), math.sin(target_angle))
    v_target_angle_2 = (math.cos(target_angle_2), math.sin(target_angle_2))
    
    direction_vec = np.array(direction_vec)
    v_target_angle = np.array(v_target_angle)
    v_target_angle_2 = np.array(v_target_angle_2)

    
    if np.linalg.norm(direction_vec) == 0 or np.linalg.norm(v_target_angle) == 0 or np.linalg.norm(v_target_angle_2) == 0:
        return target_angle
    
    cos_theta_1 = np.dot(direction_vec, v_target_angle) / (np.linalg.norm(direction_vec)*np.linalg.norm(v_target_angle))
    cos_theta_1 = np.clip(cos_theta_1, -1.0, 1.0)  

    cos_theta_2 = np.dot(direction_vec, v_target_angle_2) / (np.linalg.norm(direction_vec)*np.linalg.norm(v_target_angle_2))
    cos_theta_2 = np.clip(cos_theta_2, -1.0, 1.0)  



    # 计算两种调整方式
    # delta1 = target_angle - alpha
    # delta2 = target_angle_2 - alpha
    # print(delta1, delta2)

    # 计算两种调整方式
    # delta1 = (target_angle - alpha) % 360
    # delta2 = delta1 - 360
    
    # 选择绝对值最小的调整量
    return target_angle if np.abs(np.arccos(cos_theta_1)) < np.abs(np.arccos(cos_theta_2)) else target_angle_2

def process_points(robot_pose, vertices, points):
    # 步骤1：获取长边信息
    edge1, edge2 = calculate_long_edges(vertices)
    line1, line2 = edge1[2], edge2[2]
    angle1, angle2 = edge1[1], edge2[1]
    
    results = []
    for point in points:
        # 步骤2：计算当前点方向角
        dx = point[0] - robot_pose[0][0]
        dy = point[1] - robot_pose[0][1]
        # alpha = math.degrees(math.atan2(dy, dx)) % 360
        alpha = math.degrees(math.atan2(dy, dx))
        
        # 步骤3：确定最近长边
        dist1 = distance_point_to_line(point, line1)
        dist2 = distance_point_to_line(point, line2)
        target_angle = angle1 if dist1 < dist2 else angle2
        
        # 步骤4：计算调整量
        direction_vec = (dx, dy)
        alpha = adjust_angle(direction_vec, target_angle)
        results.append((point, alpha))
    
    return results

def show(points, arrow_length=0.5):
    """
    绘制坐标点及角度箭头
    参数：
        points: 包含坐标和角度的列表，格式[((x0,y0), α0), ((x1,y1), α1), ...]
        arrow_length: 箭头长度比例（相对于坐标轴范围）
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # 提取所有坐标点
    coordinates = np.array([p[0] for p in points])
    x_min, x_max = coordinates[:,0].min(), coordinates[:,0].max()
    y_min, y_max = coordinates[:,1].min(), coordinates[:,1].max()
    
    # 计算动态箭头长度（基于坐标范围）
    axis_range = max(x_max-x_min, y_max-y_min) * 0.2
    scale_factor = arrow_length * axis_range

    # 绘制每个点及角度箭头
    for (x, y), alpha in points:
        # 绘制坐标点[5](@ref)
        plt.scatter(x, y, c='red', s=80, edgecolor='black', zorder=3)
        
        # 计算箭头方向向量[2,8](@ref)
        dx = scale_factor * np.cos(np.deg2rad(alpha))
        dy = scale_factor * np.sin(np.deg2rad(alpha))
        
        # 绘制角度箭头[1,6](@ref)
        ax.annotate(
            '', 
            xytext=(x, y),  # 起点
            xy=(x+dx, y+dy),  # 终点
            arrowprops=dict(
                arrowstyle='->',
                linewidth=2,
                color='blue',
                mutation_scale=20,
                shrinkA=0,  # 取消起点收缩
                shrinkB=0   # 取消终点收缩
            ),
            zorder=2
        )
        
        # 添加角度文本标注[3,8](@ref)
        text_x = x + dx * 1.2
        text_y = y + dy * 1.2
        plt.text(text_x, text_y, 
                f'{alpha}°', 
                fontsize=10, 
                color='darkgreen',
                ha='center', 
                va='center')

    # 设置坐标轴
    plt.grid(linestyle='--', alpha=0.7)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Points with Directional Arrows')
    plt.axis('equal')  # 等比例坐标轴
    plt.show()
show_points = generate_all_serach_points(vertices, 0.5)
points_with_alpha = process_points(robot_pose, vertices, show_points)
for (point, alpha) in points_with_alpha:
    print(f"Point: {point}, Adjusted Angle: {alpha:.2f}°")
show(points_with_alpha)
     
