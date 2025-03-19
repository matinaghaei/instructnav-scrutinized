import numpy as np
from habitat.utils.visualizations import maps
from scipy.ndimage import grey_dilation


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def generate_image(segmentation, info):
    top_down_map = draw_top_down_map(info, segmentation.shape[0])
    output_im = np.concatenate((segmentation, top_down_map), axis=1)
    return output_im


def draw_on_grid(grid, points):
    grid = np.zeros_like(grid)
    for x, y in points:
        grid[x, y] = 1
    return grid


def dilate(grid, radius=1):
    kernel_size = 2 * radius + 1
    return grey_dilation(grid, size=(kernel_size, kernel_size))
