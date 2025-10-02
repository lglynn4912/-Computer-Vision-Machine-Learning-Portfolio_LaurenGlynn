import cv2
import os
import imageio
import numpy as np
from PIL import Image


class DataParams:
    def __init__(self, out_dir, data_dir, frame_ids):
        self.out_dir = out_dir
        self.data_dir = data_dir
        self.frame_ids = frame_ids

    def gen_data_fname(self, frame_id):
        return f'{self.data_dir}/frame{frame_id + 1}.png'

    def gen_out_fname(self, frame_id):
        return f'{self.out_dir}/frame{frame_id + 1}.png'

    def get_video_path(self):
        return os.path.join(self.out_dir, 'video.mp4')


def trackingTester(data_params, tracking_params):
    # Perform object tracking and annotation for each frame
    for frame_id in data_params.frame_ids:
        frame_path = data_params.gen_data_fname(frame_id)
        frame = imageio.imread(frame_path)

        # Track object using histogram comparison
        score = trackObject(frame, tracking_params['rect'], tracking_params['bin_n'])

        # Draw a box around the target in the frame
        annotated_frame = annotateFrame(frame, tracking_params['rect'])

        # Save the annotated frame as a PNG file
        save_annotated_frame(data_params.out_dir, annotated_frame, frame_id)

    # Generate a video file containing all the annotated frames
    generateVideo(data_params)


def trackObject(frame, target_region, bins):
    # Calculate the histogram of the target region in the frame
    x, y, w, h = target_region
    roi_frame = frame[y:y + h, x:x + w]
    hist_target = cv2.calcHist([roi_frame], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_target, hist_target)

    # Calculate the histogram of the current frame
    hist_frame = cv2.calcHist([frame], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_frame, hist_frame)

    # Compare the histograms using histogram comparison methods
    score = cv2.compareHist(hist_target, hist_frame, cv2.HISTCMP_CORREL)
    return score


def annotateFrame(frame, target_region):
    # Draw a rectangle around the target region
    x, y, w, h = target_region
    annotated_frame = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
    return annotated_frame


def save_annotated_frame(out_dir, annotated_frame, frame_number):
    file_path = os.path.join(out_dir, f"frame{frame_number + 1}.png")
    cv2.imwrite(file_path, annotated_frame)
    print(f"Saved annotated frame {frame_number + 1} to: {file_path}")


def generateVideo(data_params):
    video_path = data_params.get_video_path()  # Use the method from DataParams
    os.makedirs(data_params.out_dir, exist_ok=True)
    writer = imageio.get_writer(video_path, fps=24)  # Define fps as needed

    for frame_id in data_params.frame_ids:
        img_path = data_params.gen_out_fname(frame_id)
        writer.append_data(imageio.imread(img_path))

    writer.close()
    print(f"Video created successfully at {video_path}")