import numpy as np
import matplotlib.pyplot as plt


# 生成随机数据点
data_points = np.random.uniform(-10, 10, size=(100, 2))
true_centroid = np.mean(data_points, axis=0)
start_point = np.array([-20.0, 20.0])    # 起始点

class Sgd:
    """ 随机梯度下降算法(sthotastic gradient descent,SGD)的实现 """
    def __init__(self, learning_rate = 0.1, epochs = 30,):
        # 算法超参数
        self.learning_rate = learning_rate  #学习率
        self.epochs = epochs    #训练轮次

    def fit(self, data_points: np.ndarray, initial_point: np.ndarray) -> np.ndarray:
        current_point = initial_point.copy()    #初始化：起点即为当前点
        tr = [current_point.copy()]     #记录优化路径
        for _ in range(self.epochs):
            selected_point = data_points[np.random.randint(0, len(data_points))]
            gradient = 2 * (current_point - selected_point)
            current_point -= self.learning_rate * gradient
            tr.append(current_point.copy())
        tr = np.array(tr)
        return tr
    

class Mbgd:
    """ 小批量梯度下降算法(mini-batch gradient desent,MBGD)的实现 """
    def __init__(self, learning_rate = 0.1, epochs = 30, batch_size = 10):
        self.learning_rate = learning_rate
        self.epochs = epochs  
        self.batch_size = batch_size

    def fit(self, data_points: np.ndarray, initial_point: np.ndarray) -> np.ndarray:
        current_point = initial_point.copy()
        tr = [current_point.copy()]
        for _ in range(self.epochs):
            batch_indices = np.random.choice(len(data_points), self.batch_size, replace=False)
            batch_points = data_points[batch_indices]
            gradient = 2 * np.mean(current_point - batch_points, axis=0)
            current_point -= self.learning_rate * gradient
            tr.append(current_point.copy())
        tr = np.array(tr)
        return tr
        

if __name__ == '__main__':
    sgd = Sgd(learning_rate=0.1, epochs=30)
    mbgd = Mbgd(learning_rate=0.1, epochs=30)

    tr_SGD = sgd.fit(data_points, start_point)
    tr_MBGD = mbgd.fit(data_points, start_point)
    
    d_SGD = np.linalg.norm(tr_SGD - true_centroid, axis=1)
    d_MBGD = np.linalg.norm(tr_MBGD - true_centroid, axis=1)


# --- 可视化部分 ---

# 图 1: 两种优化路径的可视化
plt.figure(figsize=(7, 7))
plt.scatter(data_points[:, 0], data_points[:, 1], alpha=0.4, label='Data Points')
plt.plot(tr_SGD[:, 0], tr_SGD[:, 1], 'b-', label='SGD Path')
plt.plot(tr_MBGD[:, 0], tr_MBGD[:, 1], 'g-.', label='MBGD Path')
plt.scatter(tr_SGD[0, 0], tr_SGD[0, 1], c='black', s=120, zorder=5, label='Start')
plt.scatter(tr_SGD[-1, 0], tr_SGD[-1, 1], c='blue', s=120, zorder=5, label='SGD End')
plt.scatter(tr_MBGD[-1, 0], tr_MBGD[-1, 1], c='green', s=120, zorder=5, label='MBGD End')
plt.scatter(true_centroid[0], true_centroid[1], c='purple', s=200, marker='*', zorder=5, label='True Centroid')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# 图 2: 两种优化方法的收敛速度比较
plt.figure(figsize=(10, 6))
plt.plot(d_SGD, marker='o', linestyle='-', label='SGD')
plt.plot(d_MBGD, marker='^', linestyle='-.', label='MBGD')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Euclidean Distance', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
