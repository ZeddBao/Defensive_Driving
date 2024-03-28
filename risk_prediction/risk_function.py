import numpy as np
import torch
import matplotlib.pyplot as plt


# def radial_basis_function(x, y, alpha=2.0):
#     distance = np.sqrt(x**2 + y**2)
#     return np.exp(-alpha * distance)

# def inverse_distance_weighting_function(x, y, alpha=2.0):
#     distance = np.sqrt(x**2 + y**2)
#     return 1 / (1 + alpha * distance)

# def logistic_function(x, y, k=2.0):
#     if np.all(x == 0) and np.all(y == 0):
#         return 1.0
#     distance = np.sqrt(x**2 + y**2)
#     return 1 / (1 + np.exp(k * (distance - 1)))

def radial_basis_function(distance, alpha=0.1):
    return torch.exp(-alpha * distance)

def inverse_distance_weighting_function(distance, alpha=0.5):
    return 1 / (1 + alpha * distance)

def logistic_function(distance, k=0.5):
    return 1 / (1 + torch.exp(k * (distance - 1)))


if __name__ == '__main__':
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    x, y = np.meshgrid(x, y)
    x, y = torch.tensor(x), torch.tensor(y)
    distance = torch.sqrt(x**2 + y**2)

    z_rbf = np.array(radial_basis_function(distance))
    z_idw = np.array(inverse_distance_weighting_function(distance))
    z_logistic = np.array(logistic_function(distance))

    fig = plt.figure(figsize=(12, 10))

    # Radial Basis Function
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(x, y, z_rbf, cmap='viridis')
    ax1.set_title('Radial Basis Function')

    # Inverse Distance Weighting Function
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_surface(x, y, z_idw, cmap='viridis')
    ax2.set_title('Inverse Distance Weighting Function')

    # Logistic Function
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_surface(x, y, z_logistic, cmap='viridis')
    ax3.set_title('Logistic Function')


    # 设置标签
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Output value')

    plt.tight_layout()
    plt.show()