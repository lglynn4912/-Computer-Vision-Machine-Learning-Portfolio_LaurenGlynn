import argparse

from helpers import ImageClicker
from hw4_challenge1 import backwardWarpImg, computeHomography
from runTests import run_tests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def runHw4():
    # runHw4 is the "main" interface that lets you execute all the 
    # walkthroughs and challenges in this homework. It lists a set of 
    # functions corresponding to the problems that need to be solved.
    #
    # Note that this file also serves as the specifications for the functions 
    # you are asked to implement. In some cases, your submissions will be 
    # auto-graded.  Thus, it is critical that you adhere to all the specified 
    # function signatures.
    #
    # Before your submission, make sure you can run runHw4('all') 
    # without any error.
    #
    # Usage:
    # python runHw4.py                  : list all the registered functions
    # python runHw4.py 'function_name'  : execute a specific test
    # python runHw4.py 'all'            : execute all the registered functions
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {
        'honesty': honesty, 
        'challenge1a': challenge1a,
        'challenge1b': challenge1b,
        'challenge1c': challenge1c,
        'challenge1d': challenge1d, 
        'challenge1e': challenge1e, 
        'challenge1f': challenge1f,
    }
    run_tests(args.function_name, fun_handles)

# Academic Honesty Policy
def honesty():
    from signAcademicHonestyPolicy import sign_academic_honesty_policy
    # Type your full name and uni (both in string) to state your agreement 
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy('Lauren Glynn', 'pp117')

# Tests for Challenge 1: Panoramic Photo App

# Test homography
def challenge1a():
    from helpers import ImageClicker
    from hw4_challenge1 import computeHomography, applyHomography, showCorrespondence

    orig_img = Image.open('data/portrait.png')
    orig_img = np.array(orig_img)

    warped_img = Image.open('data/portrait_transformed.png')
    warped_img = np.array(warped_img)

    # Choose 4 corresponding points
    # src_pts_nx2 and dest_pts_nx2 are the coordinates of corresponding points
    # of the two images, respectively. src_pts_nx2 and dest_pts_nx2
    # are nx2 matrices, where the first column contains
    # the x coordinates and the second column contains the y coordinates.
    # Either specify them here or use the ImageClicker class to select them
    # src_pts_nx2 = np.array([[xs1, ys1], [xs2, ys2], [xs3, ys3], [xs4, ys4]])
    # dest_pts_nx2 = np.array([[xd1, yd1], [xd2, yd2], [xd3, yd3], [xd4, yd4]])
    clicker = ImageClicker('data/portrait.png', 4)
    clicker.run()
    src_pts_nx2 = clicker.get_points()
    print("Source image points", src_pts_nx2)
    clicker = ImageClicker('data/portrait_transformed.png', 4)
    clicker.run()
    dest_pts_nx2 = clicker.get_points()
    print("Destination image points", dest_pts_nx2)

    # H_3x3, a 3x3 matrix, is the estimated homography that
    # transforms src_pts_nx2 to dest_pts_nx2.
    H_3x3 = computeHomography(src_pts_nx2, dest_pts_nx2)

    # Choose another set of points on orig_img for testing.
    # test_pts_nx2 should be an nx2 matrix, where n is the number of points, the
    # first column contains the x coordinates and the second column contains
    # the y coordinates.
    # test_pts_nx2 = np.array([[xt1, yt1], [xt2, yt2], [xt3, yt3], [xt4, yt4]])

    clicker = ImageClicker('data/portrait.png', 4)
    clicker.run()
    test_pts_list = clicker.get_points()
    print("Test points on original image:", test_pts_list)
    test_pts_nx2 = np.array(src_pts_nx2)

    # Apply homography
    dest_pts_nx2 = applyHomography(H_3x3, test_pts_nx2)

    # Verify homography
    result_img = showCorrespondence(orig_img, warped_img, test_pts_nx2, dest_pts_nx2)

    # Save the result image
    result_img = Image.fromarray(result_img.astype(np.uint8))
    result_img.save('outputs/homography_result.png')


# Test wrapping
def invert_mask(mask):
    """Invert a binary mask."""
    if isinstance(mask, Image.Image):
        mask = np.array(mask)  # Convert PIL Image to NumPy array if necessary
    return 255 - mask

def challenge1b():
    # # Load images and convert them to normalized NumPy arrays
    bg_img = np.array(Image.open('data/Osaka.png')) / 255.0
    portrait_img = np.array(Image.open('data/portrait_small.png')) / 255.0

    clicker = ImageClicker('data/Osaka.png', 4)
    clicker.run()
    bg_pts = clicker.get_points()
    print("Background points", bg_pts)
    clicker = ImageClicker('data/portrait_small.png', 4)
    clicker.run()
    portrait_pts = clicker.get_points()
    print("Portrait points", bg_pts)
    # Compute homography
    H_3x3 = computeHomography(portrait_pts, bg_pts)

    # Warp the image
    dest_img, mask = backwardWarpImg(portrait_img, np.linalg.inv(H_3x3), bg_img.shape[:2])

    # Ensure the mask is compatible for compositing
    mask_3d = np.stack([mask] * 3, axis=-1) / 255.0

    # Composite images
    result = bg_img * (1 - mask_3d) + dest_img * mask_3d
    result = Image.fromarray((result * 255).astype(np.uint8))
    result.save('outputs/Van_Gogh_in_Osaka.png')

    plt.imshow(result)
    plt.title('Van Gogh in Osaka')
    plt.show()


# Test RANSAC -- outlier rejection
def challenge1c():
    from helpers import genSIFTMatches
    from hw4_challenge1 import showCorrespondence, runRANSAC
    img_src = np.array(Image.open('data/mountain_left.png').convert('RGB'))
    img_dst = np.array(Image.open('data/mountain_center.png').convert('RGB'))

    xs, xd = genSIFTMatches(img_src, img_dst)
    xs = np.flip(xs, axis=1)
    xd = np.flip(xd, axis=1)
    # xs and xd are the centers of matched frames
    # xs and xd are nx2 matrices, where the first column contains the x
    # coordinates and the second column contains the y coordinates

    before_img = showCorrespondence(img_src, img_dst, xs, xd)
    before_img = Image.fromarray((before_img * 255).astype(np.uint8))
    before_img.save('outputs/before_ransac.png')

    plt.figure()
    plt.imshow(before_img)
    plt.title('Before RANSAC')
    plt.show()


    # Use RANSAC to reject outliers
    ransac_n = 100  # Max number of iterations
    ransac_eps = 2.0 #alignment

    inliers_id, _ = runRANSAC(xs, xd, ransac_n, ransac_eps)
    after_img = showCorrespondence(img_src, img_dst, xs[inliers_id, :], xd[inliers_id, :])
    after_img = Image.fromarray((after_img * 255).astype(np.uint8))
    after_img.save('outputs/after_ransac.png')

    plt.figure()
    plt.imshow(after_img)
    plt.title('After RANSAC')
    plt.show()

# Test image blending
def challenge1d():
    from hw4_challenge1 import blendImagePair
    fish = Image.open('data/escher_fish.png').convert('RGB')
    fish_mask = Image.open('data/escher_fish.png').convert('L')

    horse = Image.open('data/escher_horsemen.png').convert('RGB')
    horse_mask = Image.open('data/escher_horsemen.png').convert('L')

    # blended_result = blendImagePair([fish], [fish_mask], horse, horse_mask, 'blend')
    blended_result = blendImagePair(fish, fish_mask, horse, horse_mask, 'blend')
    blended_result.save('outputs/blended_result.png')

    # overlay_result = blendImagePair([fish], [fish_mask], horse, horse_mask, 'overlay')
    overlay_result = blendImagePair(fish, fish_mask, horse, horse_mask, 'overlay')
    overlay_result.save('outputs/overlay_result.png')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(fish); axs[0, 0].set_title('Escher Fish')
    axs[0, 1].imshow(horse); axs[0, 1].set_title('Escher Horse')
    axs[1, 0].imshow(blended_result); axs[1, 0].set_title('Blended')
    axs[1, 1].imshow(overlay_result); axs[1, 1].set_title('Overlay')
    plt.show()

# Test image stitching
def challenge1e():
    from hw4_challenge1 import stitchImg
    # stitch three images
    # Load images as numpy arrays
    img_center = np.array(Image.open('data/mountain_center.png'))
    img_left = np.array(Image.open('data/mountain_left.png'))
    img_right = np.array(Image.open('data/mountain_right.png'))

    # Stitch images
    stitched_img = stitchImg(img_center, img_left, img_right)

    # Display and save the stitched image
    plt.imshow(stitched_img)
    plt.title('Stitched Image')
    plt.savefig('outputs/stitched_img.png')

# Test image stitching
def challenge1f():
    from hw4_challenge1 import stitchImg
    # stitch three images
    # Load images as numpy arrays
    img_center = np.array(Image.open('data/image0 (2).jpeg'))
    img_left = np.array(Image.open('data/image1.jpeg'))
    img_right = np.array(Image.open('data/image2.jpeg'))

    # Stitch images
    stitched_img = stitchImg(img_center, img_left, img_right)

    # Display and save the stitched image
    plt.imshow(stitched_img)
    plt.title('Original Stitched Image')
    plt.savefig('outputs/Original Stitched Image.png')

if __name__ == '__main__':
    runHw4()