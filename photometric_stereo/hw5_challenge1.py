from PIL import Image
import numpy as np
from typing import Union, Tuple, List
from skimage import filters, measure
import sys
import numpy as np
from typing import List, Tuple

# def findSphere(img: np.ndarray) -> Tuple[np.ndarray, float]:
#     # Find the center and radius of the sphere
#     # Input:
#     #   img - the image of the sphere
#     # Output:
#     #   center - 2x1 vector of the center of the sphere
#     #   radius - radius of the sphere
#
#     # get grey_scale
#     grey_img = img.mean(axis=2) if img.ndim == 3 else img
#
#     # otsu + re-label
#     threshold_value = filters.threshold_otsu(grey_img)
#     binary_img = grey_img > threshold_value
#     labeled_img = measure.label(binary_img)
#
#     # get properties of labeled img
#     properties = measure.regionprops(labeled_img)
#
#     # get the largest property region using index (avoid a loop)
#     max_area_index = np.argmax([property.area for property in properties])
#
#     # find centroid + diameter (2 * radius) of largest property region
#     centroid = properties[max_area_index].centroid
#     radius = properties[max_area_index].equivalent_diameter / 2
#
#     return centroid, radius
#     # raise NotImplementedError

def findSphere(img: np.ndarray) -> Tuple[np.ndarray, float]:
    grey_img = img.mean(axis=2) if img.ndim == 3 else img
    threshold_value = filters.threshold_otsu(grey_img)
    binary_img = grey_img > threshold_value
    labeled_img = measure.label(binary_img)
    properties = measure.regionprops(labeled_img)
    max_area_index = np.argmax([prop.area for prop in properties])
    centroid = properties[max_area_index].centroid
    radius = properties[max_area_index].equivalent_diameter / 2

    print(f"Detected Sphere Center: {centroid}, Radius: {radius}")
    return centroid, radius

def compute_normal(center: np.ndarray, radius: float, point: np.ndarray) -> np.ndarray:
    # Get x,y coordinates relative to center
    x_prime = point[0] - center[0]
    y_prime = point[1] - center[1]

    # Normalize vector - for center to point coordinates
    norm = np.sqrt(x_prime ** 2 + y_prime ** 2)
    vector_x = x_prime / norm
    vector_y = y_prime / norm

    # Get normal vector (using the radius)
    normalized_x = radius * vector_x
    normalized_y = radius * vector_y

    # Calculate the square of the z component, and check if it's non-negative
    z_squared = radius ** 2 - normalized_x ** 2 - normalized_y ** 2
    normalized_z = np.sqrt(z_squared) if z_squared >= 0 else 0

    return np.array([normalized_x, normalized_y, normalized_z])


def computeLightDirections(center: np.ndarray, radius: float, images: List[np.ndarray]) -> np.ndarray:
    # Compute the light source directions
    # Input:
    #   center - 2x1 vector of the center of the sphere
    #   radius - radius of the sphere
    #   images - list of N images
    # Output:
    #   light_dirs_5x3 - 5x3 matrix of light source directions

    #   list of diff light directions
    light_dirs = []

    # go over each image
    for img in images:
        # get brightest pixel
        brightest_pixel = np.unravel_index(np.argmax(img), img.shape)

        # use normal vector to find the surface point where brightest
        normal = compute_normal(center, radius, brightest_pixel)

        # scale the normal vector - using magnitude of the brightest pixel
        magnitude = img[brightest_pixel]
        scaled_normal = normal * magnitude

        light_dirs.append(scaled_normal)

    # take list of light direction and  make a numpy array
    light_dirs_5x3 = np.array(light_dirs)

    return light_dirs_5x3

def check_light_directions(light_dirs):
    print("Light directions:\n", light_dirs)
    matrix_A = light_dirs.T @ light_dirs
    condition_number = np.linalg.cond(matrix_A)
    print("Condition number of matrix A:", condition_number)
    return condition_number


def computeMask(images: List[np.ndarray]) -> np.ndarray:
    # Compute the mask of the object
    # Input:
    #   images - list of N images
    # Output:
    #   mask - HxW binary mask

    # Convert the list of images to a NumPy array
    img_array = np.array(images)

    # get shape of the mask; get HxW dimenions
    mask_shape = images[0].shape[:2]

    # mask with zeros
    mask = np.zeros(mask_shape, dtype=np.uint8)

    # Check if each pixel is zero in all images
    mask[np.all(img_array == 0, axis=0)] = 0  # Background pixels
    mask[np.any(img_array != 0, axis=0)] = 1  # Foreground pixels

    ## check
    # print("Mask unique values:", np.unique(mask, return_counts=True))

    return mask


# def computeNormals(light_dirs: np.ndarray, images: List[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     # Compute the surface normals and albedo of the object
#     # Input:
#     #   light_dirs - Nx3 matrix of light directions
#     #   images - list of N images
#     #   mask - binary mask
#     # Output:
#     #   normals - HxWx3 matrix of surface normals
#     #   albedo_img - HxW matrix of albedo values
#
#     img_array = np.stack(images)  # Stack images into a single numpy array for easier indexing
#     height, width = mask.shape
#     normals = np.zeros((height, width, 3))
#     albedo_img = np.zeros((height, width))
#
#     print("Light directions check:", light_dirs)  # Debug: Verify light directions
#
#     # Check if the light directions matrix is singular
#     if np.linalg.cond(light_dirs.T @ light_dirs) > 1 / np.finfo(float).eps:
#         print("Warning: Light directions matrix is close to singular.")
#         return normals, albedo_img
#
#     for i in range(height):
#         for j in range(width):
#             if mask[i, j]:  # Only process foreground pixels
#                 A = light_dirs.T @ light_dirs
#                 b = light_dirs.T @ img_array[:, i, j]
#                 try:
#                     albedo = np.linalg.solve(A, b)
#                 except np.linalg.LinAlgError:
#                     albedo = np.linalg.lstsq(A, b, rcond=None)[0]
#
#                 normals[i, j] = np.linalg.pinv(A) @ light_dirs.T @ img_array[:, i, j]
#                 albedo_img[i, j] = np.linalg.norm(albedo)
#
#     print("Sample normal output:", normals[10, 10])  # Output sample normal for diagnostics
#     return normals, albedo_img




def computeNormals(light_dirs: np.ndarray, images: np.ndarray, mask: np.ndarray):
    height, width, num_images = images.shape
    normals = np.zeros((height, width, 3))
    albedo_img = np.zeros((height, width))

    A = light_dirs.T @ light_dirs

    for i in range(height):
        for j in range(width):
            if mask[i, j]:
                # Extracting all image data for pixel (i, j) across all images
                pixel_values = images[i, j, :]
                b = light_dirs.T @ pixel_values  # This now aligns properly with the dimensions

                if np.linalg.norm(b) == 0:
                    print(f"Skipping zero vector b at ({i}, {j})")
                    continue

                albedo, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                normal = np.linalg.pinv(A) @ b
                norm_length = np.linalg.norm(normal)
                if norm_length > 0:
                    normals[i, j, :] = normal / norm_length
                    albedo_img[i, j] = np.linalg.norm(albedo)
                else:
                    print(f"Zero norm at ({i}, {j}) with b: {b}")

    return normals, albedo_img



