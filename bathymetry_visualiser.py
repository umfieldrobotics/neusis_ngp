import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def visualize_bathymetry_and_rotate(npy_file1, npy_file2, gif_output_path):
    # Load the heightmap data
    heightmap1 = np.load(npy_file1)
    heightmap2 = np.load(npy_file2)
    
    # Normalize the heightmaps
    heightmap1 = (heightmap1 - np.min(heightmap1)) / (np.max(heightmap1) - np.min(heightmap1))
    heightmap2 = (heightmap2 - np.min(heightmap2)) / (np.max(heightmap2) - np.min(heightmap2))
    
    # Compute MAE and STD
    error_mae = mae(heightmap1, heightmap2)
    std1 = np.std(heightmap1)
    std2 = np.std(heightmap2)
    
    # Compute SSIM error
    heightmap1_torch = torch.tensor(heightmap1).unsqueeze(0).unsqueeze(0)
    heightmap2_torch = torch.tensor(heightmap2).unsqueeze(0).unsqueeze(0)
    ssim_error = ssim(heightmap1_torch, heightmap2_torch).item()
    
    print(f"MAE: {error_mae}")
    print(f"STD Heightmap 1: {std1}, STD Heightmap 2: {std2}")
    print(f"SSIM Error: {ssim_error}")
    
    # Create coordinate grids
    x1 = np.arange(heightmap1.shape[1])
    y1 = np.arange(heightmap1.shape[0])
    X1, Y1 = np.meshgrid(x1, y1)
    
    x2 = np.arange(heightmap2.shape[1])
    y2 = np.arange(heightmap2.shape[0])
    X2, Y2 = np.meshgrid(x2, y2)
    
    # Set up the figure and the 3D axes
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    ax1.set_title('neusis_ngp bathymetry')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Normalized Depth/Elevation')

    ax2.set_title('ground truth heightmap')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Normalized Depth/Elevation')

    # Create the surface plots
    surface1 = ax1.plot_surface(X1, Y1, heightmap1, cmap='plasma', edgecolor='none')
    surface2 = ax2.plot_surface(X2, Y2, heightmap2, cmap='plasma', edgecolor='none')

    # Function to update the view for animation, including elevation change
    def update_view(num, ax1, ax2):
        azim_angle = num
        elev_angle = 30 + 15 * np.sin(np.radians(num))  # Oscillating elevation to simulate Z-movement
        ax1.view_init(elev=elev_angle, azim=azim_angle)
        ax2.view_init(elev=elev_angle, azim=azim_angle)
        return [surface1, surface2]

    # Create animation
    ani = FuncAnimation(fig, update_view, frames=np.arange(0, 360, 2), fargs=(ax1, ax2), interval=50, blit=False)

    # Save the animation as a GIF
    ani.save(gif_output_path, writer='pillow', fps=20)

    plt.show()

# Example usage
visualize_bathymetry_and_rotate('/home/sacchin/Desktop/YIP/neusis_ngp/experiments/scene_aerial_01_/meshes/00066555.npy',
                                 '/home/sacchin/Desktop/YIP/neusis_ngp/data/scene_aerial_01_/heightmap_gt.npy',
                                 '/home/sacchin/Desktop/bathymetry_comparison.gif')
