from PIL import Image
import numpy as np
from typing import Union, Tuple, List
import cv2
from PIL import Image, ImageDraw
from scipy.signal import correlate2d


def load_image(path):
    """ Load an image from file and convert to grayscale. """
    with Image.open(path) as img:
        return np.array(img.convert('L')) / 255.0


def computeFlow(img1, img2, win_radius, template_radius, grid_MN):
    height, width = img1.shape
    grid_spacing_y = height // grid_MN[0]
    grid_spacing_x = width // grid_MN[1]
    flow_vectors = np.zeros((grid_MN[0], grid_MN[1], 2), dtype=np.float32)

    for i in range(grid_MN[0]):
        for j in range(grid_MN[1]):
            y = i * grid_spacing_y + grid_spacing_y // 2
            x = j * grid_spacing_x + grid_spacing_x // 2

            template_y1 = max(0, y - template_radius)
            template_y2 = min(height, y + template_radius + 1)
            template_x1 = max(0, x - template_radius)
            template_x2 = min(width, x + template_radius + 1)
            template = img1[template_y1:template_y2, template_x1:template_x2]

            search_y1 = max(0, y - win_radius)
            search_y2 = min(height, y + win_radius + 1)
            search_x1 = max(0, x - win_radius)
            search_x2 = min(width, x + win_radius + 1)
            search_area = img2[search_y1:search_y2, search_x1:search_x2]

            correlation = correlate2d(search_area, template, mode='same')
            dy, dx = np.unravel_index(np.argmax(correlation), correlation.shape)
            dy -= (template.shape[0] // 2)
            dx -= (template.shape[1] // 2)
            flow_vectors[i, j, :] = [dx, dy]

    return flow_vectors

def draw_arrow(draw, start_point, end_point, fill, width):
    # Calculate arrowhead size based on the length of the arrow
    arrow_length = np.linalg.norm(np.array(end_point) - np.array(start_point))
    arrow_size = max(5, int(arrow_length / 20))  # Adjust the factor as needed
    half_arrow_size = arrow_size // 2

    # Draw a line from start_point to end_point
    draw.line([start_point, end_point], fill=fill, width=width)

    # Calculate arrowhead points
    angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    arrow_x = end_point[0] - arrow_size * np.cos(angle + np.pi / 6)
    arrow_y = end_point[1] - arrow_size * np.sin(angle + np.pi / 6)
    arrow_x2 = end_point[0] - arrow_size * np.cos(angle - np.pi / 6)
    arrow_y2 = end_point[1] - arrow_size * np.sin(angle - np.pi / 6)

    # Draw triangle for arrowhead
    draw.polygon([(end_point[0], end_point[1]),
                  (arrow_x, arrow_y),
                  (arrow_x2, arrow_y2)], fill=fill)


def overlayNeedleMap(img_list: List[np.ndarray], grid_MN: Tuple[int, int], win_radius: int, template_radius: int) -> \
List[Image.Image]:
    """ Overlay optical flow arrows on a series of images. """
    # List to store images with overlayed arrows
    # List to store images with overlayed arrows
    images_with_arrows = []

    # Iterate over consecutive pairs of images
    for i in range(len(img_list) - 1):
        img1 = img_list[i]
        img2 = img_list[i + 1]

        # Compute optical flow between consecutive images
        flow = computeFlow(img1, img2, win_radius, template_radius, grid_MN)

        # Calculate grid spacing based on the dimensions of the current image
        height, width = img1.shape[:2]
        grid_spacing_y = height // grid_MN[0]
        grid_spacing_x = width // grid_MN[1]

        # Convert image to PIL format
        img_with_arrows = Image.fromarray((img1 * 255).astype(np.uint8)).convert('RGB')
        draw = ImageDraw.Draw(img_with_arrows)

        # Overlay flow arrows on the image
        for i in range(flow.shape[0]):
            for j in range(flow.shape[1]):
                y = i * grid_spacing_y + grid_spacing_y // 2
                x = j * grid_spacing_x + grid_spacing_x // 2
                dx, dy = flow[i, j]
                end_point = (int(x + dx), int(y + dy))
                if np.hypot(dx, dy) > 0.1:  # Only draw significant movements
                    draw_arrow(draw, (x, y), end_point, fill=(255, 0, 0), width=2)

        # Append image with overlayed arrows to the list
        images_with_arrows.append(img_with_arrows)

    return images_with_arrows


