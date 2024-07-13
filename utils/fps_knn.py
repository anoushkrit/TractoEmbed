#%%
# 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import time

from sklearn.neighbors import NearestNeighbors
#%%
def plot3d(pc):
    x=pc[:,0]
    y=pc[:,1]
    z=pc[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(x, y, z, c='r')

    # Set labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Scatter Plot')
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{time_str}"
    plt.savefig(f'plots/{file_name}')

def plot_knn_help(pc, color='r', ax=None, label=None):
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(x, y, z, c=color, label=label)

    # Set labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if label is not None:
        ax.legend()

    return ax
def plot_knn(nearest_neighbors, center_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate distinct colors for each cluster
    colors = plt.cm.tab20(np.linspace(0, 1, 64))
    
    for i in range(len(nearest_neighbors)):
        points = nearest_neighbors[i]
        plot_knn_help(points, color=colors[i], ax=ax, label=f'Cluster {i+1}')
    
    # plot the center points as well
    plot_knn_help(center_points, color='black', ax=ax, label='Center Points')
    
    ax.set_title('3D Scatter Plot of KNN Groups')
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data_{time_str}.png"
    plt.savefig(f'../plots/{file_name}')

def farthest_point_sampling(points, num_samples):
    """
    Perform farthest point sampling on a 3D point cloud.

    Args:
        points (numpy.ndarray): Input point cloud data of shape (N, 3).
        num_samples (int): Number of points to sample.

    Returns:
        numpy.ndarray: Sampled points of shape (num_samples, 3).
    """
    N = points.shape[0]
    sampled_points = np.zeros((num_samples, 3))
    distances = np.ones(N) * float('inf')
    farthest = np.random.randint(0, N)  # Initialize with a random point

    for i in range(num_samples):
        sampled_points[i] = points[farthest]
        dist = np.linalg.norm(points - points[farthest], axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return sampled_points

def knn(point_cloud,center_points,k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(point_cloud)
    distances, indices = knn.kneighbors(center_points)
    if k < 64: 
        nearest_neighbors = [point_cloud[indices[i]] for i in range(len(indices))]
    else:

        nearest_neighbors = point_cloud[indices]

    assigned = set()
    for i in range(len(nearest_neighbors)):
        assigned.update(indices[i])

    # Find unassigned points and assign them to the nearest group
    unassigned_indices = np.setdiff1d(np.arange(len(point_cloud)), list(assigned))
    for idx in unassigned_indices:
        distances_to_centers = np.linalg.norm(center_points - point_cloud[idx], axis=1)
        nearest_center_idx = np.argmin(distances_to_centers)
        nearest_neighbors[nearest_center_idx] = np.vstack([nearest_neighbors[nearest_center_idx], point_cloud[idx]])    
    return nearest_neighbors
#%%

def get_nearest_neighbors(pcd, num_patches, npoints_per_patch):
    center_points = farthest_point_sampling(pcd, num_patches)
    nns = knn(pcd, center_points, npoints_per_patch)
    return nns, center_points
    
def main():
    num_samples = 64  # Number of points to sample
    group_size = 32  # Number of Points in each group 
    # pth_for_pc = '/scratch/jankita.scee.iitmandi/PointBERT/PointBERT/tract_data/dvae/pc_test_1024/AF_left/AF_left_0_599469.txt'
    pth_for_pc = '/neuro/hdf_data/105HCP/AF_left_0_599469.txt'
    point_cloud = np.loadtxt(pth_for_pc, delimiter=',')
    sampled_points = farthest_point_sampling(point_cloud, num_samples)
    knn_groups = knn(point_cloud, sampled_points, group_size)
    plot_knn(knn_groups, sampled_points)

if __name__ == "__main__":
    main()
# %%
