import numpy as np
import matplotlib.pyplot as plt



data_points = np.random.uniform(-10, 10, size=(100, 2))
true_centroid = np.mean(data_points, axis=0)
current_point = np.array([-20.0, 20.0]) 


trajectory_SGD = [current_point.copy()]
trajectory_BGD = [current_point.copy()]
trajectory_MBGD = [current_point.copy()]
learning_rate = 0.1
epochs = 30


#SGD算法的实现
for epoch in range(epochs):
    selected_point = data_points[np.random.randint(0, len(data_points))]
    gradient = 2 * (current_point - selected_point)
    current_point = current_point - learning_rate * gradient
    trajectory_SGD.append(current_point.copy())
trajectory_SGD = np.array(trajectory_SGD)
# 计算每次迭代后，当前点与真实中心点的距离
distances_SGD = np.linalg.norm(trajectory_SGD - true_centroid, axis=1)

#MBGD算法的实现
mbgd_point = np.array([-20.0, 20.0]) # 起始点    
for epoch in range(epochs):
    batch_indices = np.random.choice(len(data_points), size=10, replace=False)
    batch_points = data_points[batch_indices]
    gradient = 2 * (mbgd_point - np.mean(batch_points, axis=0))
    mbgd_point = mbgd_point - learning_rate * gradient
    trajectory_MBGD.append(mbgd_point.copy())
trajectory_MBGD = np.array(trajectory_MBGD)
distances_MBGD = np.linalg.norm(trajectory_MBGD - true_centroid, axis=1)



# --- 可视化部分 ---

# 图 1: 两种优化路径的可视化
plt.figure(figsize=(7, 7))
plt.scatter(data_points[:, 0], data_points[:, 1], alpha=0.4, label='Data Points')
plt.plot(trajectory_SGD[:, 0], trajectory_SGD[:, 1], 'b-', label='SGD Path')
plt.plot(trajectory_MBGD[:, 0], trajectory_MBGD[:, 1], 'g-.', label='MBGD Path')
plt.scatter(trajectory_SGD[0, 0], trajectory_SGD[0, 1], c='black', s=120, zorder=5, label='Start')
plt.scatter(trajectory_SGD[-1, 0], trajectory_SGD[-1, 1], c='blue', s=120, zorder=5, label='SGD End')
plt.scatter(trajectory_MBGD[-1, 0], trajectory_MBGD[-1, 1], c='green', s=120, zorder=5, label='MBGD End')
plt.scatter(true_centroid[0], true_centroid[1], c='purple', s=200, marker='*', zorder=5, label='True Centroid')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# 图 2: 两种优化方法的收敛速度比较
plt.figure(figsize=(10, 6))
plt.plot(distances_SGD, marker='o', linestyle='-', label='SGD')
plt.plot(distances_MBGD, marker='^', linestyle='-.', label='MBGD')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Euclidean Distance', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
