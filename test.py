import carla
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 导入visualize目录下的predict_tick_data_9_20.pkl
with open('visualize/validation_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# 随机选一条data
data = data[np.random.randint(0, len(data))]
print('data shape:', data.shape)

# 消除多余维度
data = data.squeeze()
ego = data[40]
ego_ord_curr = ego[20, :2]
ego_pos_curr = ego[21, :2] - ego_ord_curr
print('ego:', ego_ord_curr, ego_pos_curr)

def get_transform_matrix(ego_ord_curr, ego_pos_curr):
    # 规范化 y 轴向量
    norm_ego_pos_curr = np.array(ego_pos_curr) / np.linalg.norm(ego_pos_curr)
    
    # 计算 x 轴向量（垂直于 y 轴）
    norm_ego_x_axis = np.array([-norm_ego_pos_curr[1], norm_ego_pos_curr[0]])
    
    # 创建旋转矩阵
    rotation_matrix = np.array([
        norm_ego_x_axis,    # 新 x 轴
        norm_ego_pos_curr,  # 新 y 轴
    ]).T  # 转置以匹配二维线性代数的常规布局
    
    # 创建平移向量
    translation_vector = ego_ord_curr
    
    return rotation_matrix, translation_vector

rmatrix, tvector = get_transform_matrix(ego_ord_curr, ego_pos_curr)

for i, obj in enumerate(data):
    if i == 0:
        red_color = [1, 0, 0]
        fov = obj[:, :2] - tvector
        transformed_fov = np.dot(rmatrix, fov.T)
        first = transformed_fov[:, 0].reshape(2, 1)
        transformed_fov = np.hstack((transformed_fov, first))

        plt.plot(transformed_fov[0], transformed_fov[1], color=red_color)

    elif i >= 40:
        print('car:', i)
        # 随机一个颜色
        color = np.random.rand(3)
        traj = obj[:, :2] - tvector
        transformed_traj = np.dot(rmatrix, traj.T)
        plt.plot(transformed_traj[0, :20], transformed_traj[1, :20], color=color)
        plt.plot(transformed_traj[0, 20:], transformed_traj[1, 20:], color=color, linestyle='dotted')

        
plt.show()