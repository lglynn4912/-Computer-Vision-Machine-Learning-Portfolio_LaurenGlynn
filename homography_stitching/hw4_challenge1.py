from PIL import Image, ImageOps, ImageChops
import cv2
import numpy as np
from typing import Union, Tuple, List

from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_edt
from helpers import genSIFTMatches


def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''

    src_pts_nx2 = np.array(src_pts_nx2)
    dest_pts_nx2 = np.array(dest_pts_nx2)
    n = src_pts_nx2.shape[0]
    if n < 4:
        raise ValueError("At least four points are required to compute a homography.")

    # Initialize matrix A
    A = np.zeros((2 * n, 9))
    x, y = src_pts_nx2[:, 0], src_pts_nx2[:, 1]
    xp, yp = dest_pts_nx2[:, 0], dest_pts_nx2[:, 1]
    A[0::2] = np.column_stack([-x, -y, -np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), x*xp, y*xp, xp])
    A[1::2] = np.column_stack([np.zeros(n), np.zeros(n), np.zeros(n), -x, -y, -np.ones(n), x*yp, y*yp, yp])

    # Eigenvalue decomposition
    _, V = np.linalg.eig(A.T @ A)
    h = V[:, np.argmin(np.abs(_))]

    # Reshape to 3x3 matrix
    H = h.reshape(3, 3)
    return H

def applyHomography(H_3x3: np.ndarray, src_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Apply the homography matrix to the source points.
    Arguments:
        H_3x3: the homography matrix (3x3 numpy array).
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
    Returns:
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    '''

    src_pts_homog = np.concatenate((src_pts_nx2, np.ones((len(src_pts_nx2), 1))), axis=1)

    # homography projection
    dest_pts_homog = np.dot(H_3x3, src_pts_homog.T).T

    dest_pts_nx2 = dest_pts_homog[:, :2] / dest_pts_homog[:, 2, None]

    return dest_pts_nx2

def showCorrespondence(img1: Image.Image, img2: Image.Image, pts1_nx2: np.ndarray, pts2_nx2: np.ndarray) -> Image.Image:
    '''
    Show the correspondences between the two images.
    Arguments:
        img1: the first image.
        img2: the second image.
        pts1_nx2: the coordinates of the points in the first image (nx2 numpy array).
        pts2_nx2: the coordinates of the points in the second image (nx2 numpy array).
    Returns:
        result: image depicting the correspondences.
    '''

    # Check if images are numpy arrays and convert to RGB if they have 4 channels (assume last channel is alpha)
    if img1.shape[-1] == 4:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
    if img2.shape[-1] == 4:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

    # Calculate dimensions for the result image
    result_width = img1.shape[1] + img2.shape[1]
    result_height = max(img1.shape[0], img2.shape[0])
    result = np.ones((result_height, result_width, 3), dtype=np.uint8) * 255

    # Place images in the result image
    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    result[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    # Adjust the points for the second image
    pts2_nx2_adjusted = pts2_nx2.copy()
    pts2_nx2_adjusted[:, 0] += img1.shape[1]

    # Draw lines between corresponding points
    for pt1, pt2 in zip(pts1_nx2, pts2_nx2_adjusted):
        pt1 = tuple(pt1.astype(int))
        pt2 = tuple(pt2.astype(int))
        cv2.line(result, pt1, pt2, (255, 0, 0), 2)

    return result


# function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H, dest_canvas_width_height)

def backwardWarpImg(src_img: np.ndarray, destToSrc_H: np.ndarray, canvas_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    height, width = canvas_shape
    dest_img = np.zeros((height, width, 3), dtype=np.float32)  # Assuming color image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Mesh grid for destination coordinates
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    homog_coords = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=0)

    # Apply the inverse homography matrix
    src_coords = np.dot(destToSrc_H, homog_coords)
    src_coords /= src_coords[2, :]  # Normalize

    # Interpolation coordinates
    src_x = src_coords[0, :].reshape(height, width)
    src_y = src_coords[1, :].reshape(height, width)

    # Check bounds and create mask
    valid_coords = (src_x >= 0) & (src_x < src_img.shape[1]) & (src_y >= 0) & (src_y < src_img.shape[0])
    valid_x = np.floor(src_x[valid_coords]).astype(int)
    valid_y = np.floor(src_y[valid_coords]).astype(int)
    valid_xx = xx[valid_coords]
    valid_yy = yy[valid_coords]

    # Assign values
    dest_img[valid_yy, valid_xx] = src_img[valid_y, valid_x]
    mask[valid_yy, valid_xx] = 255

    return dest_img, mask


def blendImagePair(img1: List[Image.Image], mask1: List[Image.Image], img2: Image.Image, mask2: Image.Image,
                   mode: str) -> Image.Image:
    '''
    Blend the warped images based on the masks.
    Arguments:
        img1: list of source images.
        mask1: list of source masks.
        img2: destination image.
        mask2: destination mask.
        mode: either 'overlay' or 'blend'
    Returns:
        out_img: blended image.
    '''

    # Resize images to the same dimensions
    width = min(img1.width, img2.width)
    height = min(img1.height, img2.height)
    img1 = img1.resize((width, height))
    img2 = img2.resize((width, height))

    # Convert PIL images to NumPy arrays
    img1_array = np.array(img1)
    mask1_array = np.expand_dims(np.array(mask1), axis=2)  # Convert mask1 to have 3 channels
    img2_array = np.array(img2)
    mask2_array = np.expand_dims(np.array(mask2), axis=2)  # Convert mask2 to have 3 channels

    # Apply the blending mode
    if mode == 'blend':
        blended_img = np.where(mask1_array > 0, img1_array * 0.5 + img2_array * 0.5, img2_array)
    elif mode == 'overlay':
        blended_img = np.where(mask1_array > 0, img1_array, img2_array)
    else:
        raise ValueError("Unsupported blending mode")

    return Image.fromarray(np.uint8(blended_img))


def calculateReprojectionError(H, src_pts, dest_pts):
    """ Calculate the Euclidean reprojection error for each point. """
    src_pts_hom = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
    dest_pts_hom = np.hstack([dest_pts, np.ones((dest_pts.shape[0], 1))])

    projected_pts = (H @ src_pts_hom.T).T
    projected_pts /= projected_pts[:, 2:3]  # Normalize by the last coordinate

    errors = np.linalg.norm(projected_pts[:, :2] - dest_pts[:, :2], axis=1)
    return errors

def runRANSAC(src_pt: np.ndarray, dest_pt: np.ndarray, ransac_n: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Run the RANSAC algorithm to find the inliers between the source and
    destination points.
    Arguments:
        src_pt: the coordinates of the source points (nx2 numpy array).
        dest_pt: the coordinates of the destination points (nx2 numpy array).
        ransac_n: the number of iterations to run RANSAC.
        eps: the threshold for considering a point to be an inlier.
    Returns:
        inliers_id: the indices of the inliers (kx1 numpy array).
        H: the homography matrix (3x3 numpy array).
    '''

    # Transpose src_pt and dest_pt to swap (x, y) coordinates to (y, x)
    src_pt = np.flip(src_pt, axis=1)
    dest_pt = np.flip(dest_pt, axis=1)

    max_inliers = 0
    best_H = None
    best_inliers_id = []

    for _ in range(ransac_n):
        # Randomly sample 4 points
        indices = np.random.choice(src_pt.shape[0], 4, replace=False)
        sampled_src_pts = src_pt[indices]
        sampled_dest_pts = dest_pt[indices]

        # Compute homography from these points
        H = computeHomography(sampled_src_pts, sampled_dest_pts)

        # Calculate reprojection error for all points
        errors = calculateReprojectionError(H, src_pt, dest_pt)

        # Determine inliers
        inliers_id = np.where(errors < eps)[0]

        # Update best model if current inliers are more than max found so far
        if len(inliers_id) > max_inliers:
            max_inliers = len(inliers_id)
            best_H = H
            best_inliers_id = inliers_id

    return np.array(best_inliers_id), best_H


def stitchImg(*args: np.ndarray) -> np.ndarray:
    '''
    Stitch a list of images represented as NumPy arrays.
    Arguments:
        args: a variable number of input images (as NumPy arrays).
    Returns:
        stitched_img: the stitched image as a NumPy array.
    '''
    max_height = max(img.shape[0] for img in args)
    total_width = sum(img.shape[1] for img in args)
    stitched_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    current_x = 0
    for img in args:
        height, width, _ = img.shape
        stitched_img[:height, current_x:current_x + width] = img
        current_x += width
    return stitched_img


# Load images as numpy arrays
img_center = np.array(Image.open('data/mountain_center.png'))
img_left = np.array(Image.open('data/mountain_left.png'))
img_right = np.array(Image.open('data/mountain_right.png'))

# Stitch images
stitched_img = stitchImg(img_center, img_left, img_right)

# Display stitched image
plt.imshow(stitched_img)
plt.title('Stitched Image')
plt.show()