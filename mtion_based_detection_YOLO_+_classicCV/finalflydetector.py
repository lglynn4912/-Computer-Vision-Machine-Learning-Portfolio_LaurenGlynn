import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

import cv2
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta 
import torch
import pynvml
import threading
from tqdm import tqdm
import sys
import traceback
import pandas as pd  
import math
import time
import re  #
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import gc  # 

def verify_gpu():
    """Verify GPU is working correctly"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        del x, y
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"GPU test failed: {str(e)}")
        return False

def load_previous_progress(output_directory):
    """Load previous processing progress if it exists"""
    output_path = Path(output_directory)
    tracking_file = output_path / 'video_tracking_list.csv'
    
    if tracking_file.exists():
        try:
            tracking_df = pd.read_csv(tracking_file)
            completed_videos = set(tracking_df[tracking_df['completed'] == True]['video_name'])
            print(f"Found previous progress: {len(completed_videos)} completed videos")
            return completed_videos, tracking_df
        except Exception as e:
            print(f"Could not load previous progress: {e}")
            return set(), None
    
    return set(), None

def get_videos_to_process(video_files, completed_videos, resume_mode=True):
    """Filter videos based on previous progress"""
    if not resume_mode or not completed_videos:
        return video_files
    
    videos_to_process = []
    skipped_count = 0
    
    for video_file in video_files:
        video_name = video_file.stem
        if video_name in completed_videos:
            skipped_count += 1
        else:
            videos_to_process.append(video_file)
    
    print(f"RESUME MODE: Skipping {skipped_count} completed videos, processing {len(videos_to_process)} remaining")
    return videos_to_process

def extract_video_start_timestamp_corrected(video_path):
    """Extract video start timestamp from corrected filename format"""
    video_name = Path(video_path).stem
    
    # CORRECTED Pattern: PSUPi1_MM_DD_YYYY_HH_MM[_XX]
    # Where _XX is optional file marker (01, 02, etc.)
    pattern = r'PSUPi1_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})(?:_\d+)?'
    match = re.match(pattern, video_name)
    
    if match:
        month, day, year, hour, minute = match.groups()
        
        try:
            video_start = datetime(
                year=int(year),
                month=int(month), 
                day=int(day),
                hour=int(hour),
                minute=int(minute),
                second=0  # No seconds in filename, assume start of minute
            )
            return video_start
        except ValueError as e:
            print(f"Invalid date/time in filename {video_name}: {e}")
            return None
    else:
        print(f"Filename doesn't match expected pattern: {video_name}")
        return None

def calculate_event_timestamp_corrected(video_start_time, frame_number, fps):
    """Calculate the actual timestamp when an event occurred"""
    if video_start_time is None:
        return None
    
    # Calculate seconds from start of video
    seconds_from_start = frame_number / fps
    
    # Add to video start time
    event_timestamp = video_start_time + timedelta(seconds=seconds_from_start)
    
    return event_timestamp

def extract_file_marker(video_path):
    """Extract the file marker (01, 02, etc.) from filename"""
    video_name = Path(video_path).stem
    
    # Look for file marker at end
    pattern = r'PSUPi1_\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_(\d+)'
    match = re.search(pattern, video_name)
    
    if match:
        return int(match.group(1))
    else:
        return None

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[float, float]
    area: float
    source: str
    frame_id: int

class MotionRequiredDetector:
    """Enhanced detector that requires MOVEMENT to confirm flies (not static debris)"""
    
    def __init__(self):
        # Proven detection parameters
        self.diff_threshold = 22
        self.min_area = 50
        self.max_area = 1400
        self.min_width = 8
        self.max_width = 55
        self.min_height = 8
        self.max_height = 55
        
        # Morphological operations
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        self.blur_kernel = (5, 5)
        
        # Frame buffer for motion analysis
        self.frame_buffer = deque(maxlen=5)  # Extended buffer for motion tracking
        self.background_model = None
        self.background_alpha = 0.005
        
        # MOTION TRACKING for filtering static objects
        self.detection_history = deque(maxlen=30)  # Store recent detections
        self.static_objects = set()  # Track known static debris
        
        print("MotionRequiredDetector initialized")
        
    def detect_flies(self, frame):
        """Detect flies but FILTER OUT static debris"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        self.frame_buffer.append(gray)
        
        if len(self.frame_buffer) < 3:  # Need more frames for motion analysis
            return detections
        
        # Get raw detections using proven method
        raw_detections = self._get_raw_detections(frame)
        
        # CRITICAL: Filter out static objects
        motion_validated_detections = self._filter_static_objects(raw_detections)
        
        return motion_validated_detections
    
    def _get_raw_detections(self, frame):
        """Get raw detections using proven method"""
        detections = []
        
        # Frame difference detection
        current = self.frame_buffer[-1]
        previous = self.frame_buffer[-2]
        
        diff = cv2.absdiff(current, previous)
        _, binary = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel_close)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            detection = self._validate_contour(contour, 'frame_diff')
            if detection:
                detections.append(detection)
        
        # Background subtraction detection
        if len(self.frame_buffer) >= 4:
            bg_detections = self._background_subtraction_detection(frame)
            detections.extend(bg_detections)
        
        return self._basic_filter_and_merge(detections, frame)
    
    def _background_subtraction_detection(self, frame):
        """Background subtraction for additional validation"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        if self.background_model is None:
            self.background_model = gray.astype(np.float32)
            return detections
        
        cv2.accumulateWeighted(gray, self.background_model, self.background_alpha)
        
        bg_gray = cv2.convertScaleAbs(self.background_model)
        diff = cv2.absdiff(gray, bg_gray)
        
        _, binary = cv2.threshold(diff, self.diff_threshold + 8, 255, cv2.THRESH_BINARY)
        
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel_close)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            detection = self._validate_contour(contour, 'background_sub')
            if detection:
                detection.confidence *= 0.8
                detections.append(detection)
        
        return detections
    
    def _validate_contour(self, contour, source):
        """Standard contour validation"""
        area = cv2.contourArea(contour)
        
        if not (self.min_area <= area <= self.max_area):
            return None
        
        x, y, w, h = cv2.boundingRect(contour)
        
        if not (self.min_width <= w <= self.max_width and 
                self.min_height <= h <= self.max_height):
            return None
        
        aspect_ratio = w / h if h > 0 else 0
        if not (0.3 <= aspect_ratio <= 3.5):
            return None
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
        else:
            compactness = 0
        
        # Confidence calculation
        confidence = 0.4
        
        if 70 <= area <= 400:
            confidence += 0.4
        elif 50 <= area <= 600:
            confidence += 0.3
        elif 40 <= area <= 800:
            confidence += 0.2
        
        if 0.15 <= compactness <= 0.7:
            confidence += 0.3
        elif 0.1 <= compactness <= 0.8:
            confidence += 0.2
        
        if source == 'frame_diff':
            confidence += 0.2
        
        confidence = min(confidence, 1.0)
        
        if confidence > 0.55:
            center_x = x + w // 2
            center_y = y + h // 2
            
            return Detection(
                bbox=(x, y, x + w, y + h),
                confidence=confidence,
                center=(center_x, center_y),
                area=area,
                source=source,
                frame_id=0
            )
        
        return None
    
    def _basic_filter_and_merge(self, detections, frame):
        """Basic filtering without motion analysis"""
        if not detections:
            return detections
        
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for detection in detections:
            overlap_found = False
            for existing in filtered:
                overlap = self._calculate_overlap(detection.bbox, existing.bbox)
                if overlap > 0.3:
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered.append(detection)
        
        height, width = frame.shape[:2]
        edge_margin = 15
        
        final = []
        for detection in filtered:
            x1, y1, x2, y2 = detection.bbox
            if (x1 > edge_margin and y1 > edge_margin and 
                x2 < width - edge_margin and y2 < height - edge_margin):
                final.append(detection)
        
        return final
    
    def _filter_static_objects(self, detections):
        """CRITICAL: Filter out static debris/dirt spots"""
        if not detections:
            return detections
        
        # Add current detections to history
        current_positions = [(d.center, d.confidence) for d in detections]
        self.detection_history.append(current_positions)
        
        # Need sufficient history for motion analysis
        if len(self.detection_history) < 10:
            return []  # Don't trust early detections
        
        moving_detections = []
        
        for detection in detections:
            if self._is_moving_object(detection):
                # Additional validation: check for actual movement pattern
                if self._validate_movement_pattern(detection):
                    moving_detections.append(detection)
                    # Boost confidence for confirmed moving objects
                    detection.confidence = min(1.0, detection.confidence + 0.2)
            else:
                # Mark as potential static object
                static_key = (int(detection.center[0] / 20), int(detection.center[1] / 20))  # Grid position
                self.static_objects.add(static_key)
        
        return moving_detections
    
    def _is_moving_object(self, detection):
        """Check if detection represents a moving object"""
        current_pos = detection.center
        
        # Check recent detection history
        movement_evidence = 0
        static_evidence = 0
        
        for frame_detections in list(self.detection_history)[-10:]:  # Last 10 frames
            for past_pos, past_conf in frame_detections:
                distance = math.sqrt((current_pos[0] - past_pos[0])**2 + 
                                   (current_pos[1] - past_pos[1])**2)
                
                if distance < 15:  # Very close to current position
                    static_evidence += 1
                elif 15 <= distance <= 100:  # Reasonable movement distance
                    movement_evidence += 1
        
        # Static objects appear in same location repeatedly
        if static_evidence > 5 and movement_evidence < 2:
            return False
        
        # Check if this location is known static debris
        grid_pos = (int(current_pos[0] / 20), int(current_pos[1] / 20))
        if grid_pos in self.static_objects:
            return False
        
        return True
    
    def _validate_movement_pattern(self, detection):
        """Validate that detection shows fly-like movement"""
        current_pos = detection.center
        
        # Look for movement trajectory in recent frames
        recent_positions = []
        for frame_detections in list(self.detection_history)[-5:]:
            for past_pos, past_conf in frame_detections:
                distance = math.sqrt((current_pos[0] - past_pos[0])**2 + 
                                   (current_pos[1] - past_pos[1])**2)
                if distance <= 50:  # Could be same object
                    recent_positions.append(past_pos)
        
        if len(recent_positions) < 2:
            return True  # Not enough data, allow it
        
        # Check for actual movement
        total_movement = 0
        for i in range(1, len(recent_positions)):
            dist = math.sqrt((recent_positions[i][0] - recent_positions[i-1][0])**2 + 
                           (recent_positions[i][1] - recent_positions[i-1][1])**2)
            total_movement += dist
        
        # Require minimum movement to confirm as fly
        return total_movement > 20  # Must move at least 20 pixels total
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

class MovementBasedFlyCounter:
    """Count flies based on confirmed movement patterns"""
    
    def __init__(self):
        self.confirmed_movement_detections = []
        self.all_detections = []
        self.frame_count = 0
        
    def add_detections(self, detections, frame_id):
        """Add movement-validated detections"""
        self.frame_count += 1
        
        for detection in detections:
            detection.frame_id = frame_id
            self.all_detections.append(detection)
            
            # All detections from motion detector are already movement-validated
            if detection.confidence >= 0.6:  # Lower threshold since already filtered
                self.confirmed_movement_detections.append(detection)
    
    def estimate_fly_count(self):
        """Estimate fly count using movement-based methods"""
        methods = {}
        
        # Method 1: Movement-confirmed detections
        methods['movement_confirmed'] = len(self.confirmed_movement_detections)
        
        # Method 2: Spatial clustering of moving objects
        clusters = self._cluster_moving_detections(self.confirmed_movement_detections)
        methods['spatial_clusters'] = len(clusters)
        
        # Method 3: Temporal movement analysis
        temporal_groups = self._analyze_temporal_movement(self.confirmed_movement_detections)
        methods['temporal_movement'] = len(temporal_groups)
        
        # Method 4: Conservative estimate based on sustained movement
        if self.frame_count > 0:
            movement_rate = len(self.all_detections) / self.frame_count
            if movement_rate > 0.02:  # At least 2% of frames show movement
                methods['sustained_movement'] = min(int(movement_rate * 50), 8)
            else:
                methods['sustained_movement'] = 0
        else:
            methods['sustained_movement'] = 0
        
        return methods
    
    def _cluster_moving_detections(self, detections):
        """Cluster detections that show movement"""
        if not detections:
            return []
        
        clusters = []
        used_detections = set()
        
        for i, detection in enumerate(detections):
            if i in used_detections:
                continue
            
            cluster = [detection]
            used_detections.add(i)
            
            for j, other_detection in enumerate(detections):
                if j in used_detections:
                    continue
                
                distance = math.sqrt((detection.center[0] - other_detection.center[0])**2 + 
                                   (detection.center[1] - other_detection.center[1])**2)
                # Larger clustering distance for moving objects
                if distance < 120:
                    cluster.append(other_detection)
                    used_detections.add(j)
            
            if len(cluster) >= 2:  # At least 2 movement detections
                clusters.append(cluster)
        
        return clusters
    
    def _analyze_temporal_movement(self, detections):
        """Analyze temporal patterns of movement"""
        if not detections:
            return []
        
        # Group by larger time windows for movement analysis
        frame_groups = {}
        for detection in detections:
            time_group = detection.frame_id // 90  # 3-second windows
            if time_group not in frame_groups:
                frame_groups[time_group] = []
            frame_groups[time_group].append(detection)
        
        # Only count periods with sustained movement activity
        significant_groups = [g for g in frame_groups.values() if len(g) >= 3]
        
        return significant_groups
    
    def get_movement_based_count(self):
        """Get fly count based on confirmed movement"""
        methods = self.estimate_fly_count()
        
        # Remove zero estimates
        non_zero_estimates = [v for v in methods.values() if v > 0]
        
        if not non_zero_estimates:
            return 0, methods
        
        # Use conservative estimate (lower quartile)
        non_zero_estimates.sort()
        if len(non_zero_estimates) == 1:
            conservative_count = non_zero_estimates[0]
        else:
            conservative_count = non_zero_estimates[len(non_zero_estimates) // 3]  # Lower quartile
        
        # Cap at reasonable maximum
        conservative_count = min(conservative_count, 8)
        
        return conservative_count, methods

class MotionBasedBatchProcessor:
    """Batch processor that requires actual movement to detect flies"""
    
    def __init__(self, output_directory="motion_based_batch"):
        self.output_path = Path(output_directory)
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        # YOLO validation (optional)
        self.use_yolo = False  # Disable for speed since we have motion validation
        
        print("MotionBasedBatchProcessor initialized")
    
    def save_detailed_detection_csv(self, results, video_name):
        """Save CSV with individual detection timestamps - ALWAYS save even if no detections"""
        
        # Create basic video info row even if no detections
        if 'timestamped_detections' not in results or not results['timestamped_detections']:
            # Create empty detection CSV with video metadata
            empty_data = [{
                'video_name': video_name,
                'video_start': results.get('video_start_readable', ''),
                'file_marker': results.get('file_marker', ''),
                'frame_number': None,
                'confidence': None,
                'bbox': None,
                'center': None,
                'area': None,
                'source': None,
                'seconds_from_video_start': None,
                'event_timestamp': None,
                'event_datetime_readable': None,
                'event_date': None,
                'event_time': None,
                'event_hour': None,
                'event_minute': None,
                'no_moving_flies_detected': True
            }]
            
            detections_df = pd.DataFrame(empty_data)
            csv_path = self.output_path / f"{video_name}_detections.csv"
            detections_df.to_csv(csv_path, index=False)
            
            return 0  # No detections but file was created
        
        # Normal case with detections
        detections_df = pd.DataFrame(results['timestamped_detections'])
        detections_df['video_name'] = video_name
        detections_df['video_start'] = results.get('video_start_readable', '')
        detections_df['file_marker'] = results.get('file_marker', '')
        detections_df['no_moving_flies_detected'] = False
        
        csv_path = self.output_path / f"{video_name}_detections.csv"
        detections_df.to_csv(csv_path, index=False)
        
        return len(detections_df)
        
    def process_single_video_motion_based(self, video_path):
        """Process video requiring actual movement"""
        video_path = Path(video_path)
        video_name = video_path.stem
        
        print(f"\nProcessing: {video_name}")
        
        # Create fresh detector and counter for each video
        detector = MotionRequiredDetector()
        fly_counter = MovementBasedFlyCounter()
        
        results = {
            'video_path': str(video_path),
            'video_name': video_name,
            'processing_time': 0,
            'total_detections': 0,
            'movement_confirmed_detections': 0,
            'confirmed_flies': 0,
            'frames_saved': 0,
            'method': 'motion_required',
            'estimation_methods': {}
        }
        
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                results['error'] = f"Could not open video: {video_path}"
                return results
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # ADDED: Extract video start time and file marker
            video_start = extract_video_start_timestamp_corrected(video_path)
            file_marker = extract_file_marker(video_path)
            results['video_start_timestamp'] = video_start.isoformat() if video_start else None
            results['video_start_readable'] = video_start.strftime('%Y-%m-%d %H:%M:%S') if video_start else None
            results['video_fps'] = fps
            results['file_marker'] = file_marker
            
            progress_bar = tqdm(total=total_frames, desc=f"Processing {video_name}", 
                              unit="frames", leave=False)
            
            frame_count = 0
            frames_to_save = {}
            all_detections = []  # ADDED: Collect all detections for timestamp analysis
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect flies with movement requirement
                detections = detector.detect_flies(frame)
                
                # ADDED: Set frame IDs and collect all detections
                for detection in detections:
                    detection.frame_id = frame_count
                    all_detections.append(detection)
                
                # Add to counter
                fly_counter.add_detections(detections, frame_count)
                
                # Store frames with movement-confirmed detections
                if detections and len(frames_to_save) < 30:
                    frames_to_save[frame_count] = {
                        'frame': frame.copy(),
                        'detections': detections
                    }
                
                results['total_detections'] += len(detections)
                frame_count += 1
                progress_bar.update(1)
                
                # Memory management
                if frame_count % 1000 == 0:
                    gc.collect()
            
            # Get final fly count estimate
            movement_count, estimation_methods = fly_counter.get_movement_based_count()
            
            results['confirmed_flies'] = movement_count
            results['movement_confirmed_detections'] = len(fly_counter.confirmed_movement_detections)
            results['estimation_methods'] = estimation_methods
            
            # add timestamped info, even if no detections
            if video_start:
                # add timestamp info to results
                if all_detections:
                    timestamped_detections = []
                    event_times = []
                    
                    for detection in all_detections:
                        event_time = calculate_event_timestamp_corrected(video_start, detection.frame_id, fps)
                        if event_time:
                            event_times.append(event_time)
                            
                            detection_info = {
                                'frame_number': detection.frame_id,
                                'confidence': detection.confidence,
                                'bbox': detection.bbox,
                                'center': detection.center,
                                'area': detection.area,
                                'source': detection.source,
                                'seconds_from_video_start': detection.frame_id / fps,
                                'event_timestamp': event_time.isoformat(),
                                'event_datetime_readable': event_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                'event_date': event_time.strftime('%Y-%m-%d'),
                                'event_time': event_time.strftime('%H:%M:%S.%f')[:-3],
                                'event_hour': event_time.hour,
                                'event_minute': event_time.minute,
                            }
                            timestamped_detections.append(detection_info)
                    
                    results['timestamped_detections'] = timestamped_detections
                    
                    # Add event summary
                    if event_times:
                        results['first_fly_event'] = min(event_times).isoformat()
                        results['last_fly_event'] = max(event_times).isoformat()
                        results['fly_activity_duration_seconds'] = (max(event_times) - min(event_times)).total_seconds()
                        
                        # Time distribution analysis
                        hours = [t.hour for t in event_times]
                        results['fly_activity_hours'] = list(set(hours))
                        results['peak_activity_hour'] = max(set(hours), key=hours.count) if hours else None
                else:
                    # when no detections found - still record video was processed
                    results['timestamped_detections'] = []
                    results['first_fly_event'] = None
                    results['last_fly_event'] = None
                    results['fly_activity_duration_seconds'] = 0
                    results['fly_activity_hours'] = []
                    results['peak_activity_hour'] = None
            
            # Save representative frames (only if movement detected)
            if movement_count > 0 and frames_to_save:
                frames_saved = self._save_movement_frames(frames_to_save, video_name)
                results['frames_saved'] = frames_saved
            
            # save detection CSV - even for videos with no flies
            csv_rows = self.save_detailed_detection_csv(results, video_name)
            results['detection_csv_rows'] = csv_rows
            
            progress_bar.close()
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Error processing {video_name}: {e}")
        
        finally:
            if 'cap' in locals():
                cap.release()
            
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        results['total_frames'] = frame_count
        results['fps'] = frame_count / processing_time if processing_time > 0 else 0
        
        print(f"Completed {video_name}: {results['confirmed_flies']} MOVING flies detected, "
              f"{results['movement_confirmed_detections']} movement confirmations, "
              f"{processing_time:.1f}s, {results['fps']:.1f} fps")
        
        # Add processing status for tracking
        results['processed_successfully'] = True
        results['video_processed'] = True
        
        return results
    
    def _save_movement_frames(self, frames_to_save, video_name, max_frames=15):
        """Save frames showing confirmed movement"""
        frames_saved = 0
        
        # Sort frames by number of detections (most activity first)
        frame_activity = {}
        for frame_id, frame_data in frames_to_save.items():
            frame_activity[frame_id] = len(frame_data['detections'])
        
        # Get most active frames
        top_frames = sorted(frame_activity.items(), key=lambda x: x[1], reverse=True)
        top_frames = top_frames[:max_frames]
        
        # Save frames with annotations
        for frame_id, _ in top_frames:
            if frame_id in frames_to_save:
                frame_data = frames_to_save[frame_id]
                annotated_frame = frame_data['frame'].copy()
                
                # Draw movement-confirmed detections
                for detection in frame_data['detections']:
                    x1, y1, x2, y2 = detection.bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'MOVING: {detection.confidence:.2f}', 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                output_file = self.output_path / f"{video_name}_moving_fly_{frame_id}.jpg"
                success = cv2.imwrite(str(output_file), annotated_frame)
                if success:
                    frames_saved += 1
        
        return frames_saved

def process_folder_motion_based(folder_path, output_directory):
    """Process folder using motion-required fly detection"""
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return [], {'error': f'Folder not found: {folder_path}'}
    
    # Find video files
    video_extensions = {'.h264', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}  # Add if needed .h264
    video_files = []
    for ext in video_extensions:
        video_files.extend(folder_path.glob(f"*{ext}"))
        video_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"No video files found in {folder_path}")
        return [], {'error': f'No video files found in {folder_path}'}
    
    print(f"BATCH FLY DETECTION")
    print(f"Found {len(video_files)} total video files")

    # Check for previous progress
    output_path = Path(output_directory)
    output_path.mkdir(exist_ok=True, parents=True)

    completed_videos, previous_tracking = load_previous_progress(output_directory)

    # # Filter videos based on resume
    videos_to_process = get_videos_to_process(video_files, completed_videos, resume_mode=True)

    print(f"Output directory: {output_directory}")

    # Create/update video tracking list
    if previous_tracking is not None:
        video_tracking = previous_tracking.to_dict('records')
        # Add any new videos
        existing_names = set(previous_tracking['video_name'])
        for video_file in video_files:
            if video_file.stem not in existing_names:
                video_tracking.append({
                    'video_path': str(video_file),
                    'video_name': video_file.stem,
                    'attempted': False,
                    'completed': False,
                    'error': None,
                    'processing_time': None,
                    'confirmed_flies': None
                })
    else:
        video_tracking = []
        for video_file in video_files:
            video_tracking.append({
            'video_path': str(video_file),
            'video_name': video_file.stem,
            'attempted': False,
            'completed': False,
            'error': None,
            'processing_time': None,
            'confirmed_flies': None
        })
    
    # Save initial video list
    output_path = Path(output_directory)
    output_path.mkdir(exist_ok=True, parents=True)
    
    tracking_df = pd.DataFrame(video_tracking)
    tracking_df.to_csv(output_path / 'video_tracking_list.csv', index=False)
    print(f"Video tracking list saved: {len(video_files)} videos to process")
    
    # Initialize batch processor
    batch_processor = MotionBasedBatchProcessor(output_directory)
    
    # Process videos
    all_results = []
    total_start_time = time.time()
    
    for i, video_file in enumerate(videos_to_process):
        video_name = video_file.stem
        print(f"Processing {i+1}/{len(video_files)}: {video_name}")
        
        # Mark as attempted
        video_tracking[i]['attempted'] = True
        
        try:
            result = batch_processor.process_single_video_motion_based(video_file)
            all_results.append(result)
            
            # Mark as completed
            video_tracking[i]['completed'] = True
            video_tracking[i]['processing_time'] = result.get('processing_time', 0)
            video_tracking[i]['confirmed_flies'] = result.get('confirmed_flies', 0)
            
        except Exception as e:
            print(f"ERROR processing {video_name}: {e}")
            video_tracking[i]['error'] = str(e)
            
            # Still add a result with error info
            error_result = {
                'video_path': str(video_file),
                'video_name': video_name,
                'error': str(e),
                'processing_time': 0,
                'confirmed_flies': 0
            }
            all_results.append(error_result)
        
        # Update tracking file every 10 videos
        if (i + 1) % 10 == 0:
            tracking_df = pd.DataFrame(video_tracking)
            tracking_df.to_csv(output_path / 'video_tracking_list.csv', index=False)
            print(f"Progress checkpoint: {i+1}/{len(video_files)} videos processed")
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final tracking update
    tracking_df = pd.DataFrame(video_tracking)
    tracking_df.to_csv(output_path / 'video_tracking_list.csv', index=False)
    
    total_time = time.time() - total_start_time
    
    # Create summary
    summary = create_motion_based_summary(all_results, total_time)
    
    # ADDED: Add tracking summary to main summary
    attempted_count = sum(1 for v in video_tracking if v['attempted'])
    completed_count = sum(1 for v in video_tracking if v['completed'])
    error_count = sum(1 for v in video_tracking if v['error'] is not None)
    
    summary['tracking_summary'] = {
        'total_videos_found': len(video_files),
        'videos_attempted': attempted_count,
        'videos_completed': completed_count,
        'videos_with_errors': error_count,
        'completion_rate': f"{completed_count}/{len(video_files)} ({completed_count/len(video_files)*100:.1f}%)"
    }
    
    # Save results
    with open(output_path / 'motion_based_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_path / 'motion_based_summary.csv', index=False)
    
    # Print summary
    print_motion_based_summary(summary, len(video_files))
    
    return all_results, summary

def create_motion_based_summary(results, total_time):
    """Create summary for motion-based processing"""
    # Count ALL videos that were attempted (not just successful ones)
    total_attempted = len(results)
    successful_results = [r for r in results if 'error' not in r]
    videos_with_detections = [r for r in successful_results if r['confirmed_flies'] > 0]
    videos_without_detections = [r for r in successful_results if r['confirmed_flies'] == 0]
    
    if not results:
        return {'error': 'No videos were processed'}
    
    summary = {
        'total_videos_attempted': total_attempted,
        'successful_videos': len(successful_results),
        'failed_videos': total_attempted - len(successful_results),
        'videos_with_moving_flies': len(videos_with_detections),
        'videos_without_moving_flies': len(videos_without_detections),
        'total_processing_time': total_time,
        'total_moving_flies': sum(r['confirmed_flies'] for r in successful_results),
        'total_movement_detections': sum(r['movement_confirmed_detections'] for r in successful_results),
        'total_frames_saved': sum(r['frames_saved'] for r in successful_results),
        'total_detection_csvs_created': len([r for r in successful_results if r.get('detection_csv_rows', 0) >= 0]),
        'average_flies_per_video': sum(r['confirmed_flies'] for r in successful_results) / len(successful_results) if successful_results else 0,
        'average_processing_time': sum(r['processing_time'] for r in successful_results) / len(successful_results) if successful_results else 0,
        'average_fps': sum(r['fps'] for r in successful_results) / len(successful_results) if successful_results else 0
    }
    
    return summary


def filter_files_after_date(video_files, after_month, after_day):
    """Filter files that come after a specific month/day (any year)"""
    
    # Sort files chronologically by name
    sorted_files = sorted(video_files, key=lambda x: x.name)
    
    filtered = []
    for video_file in sorted_files:
        # Extract month and day from filename
        # Pattern: PSUPi1_MM_DD_YYYY_...
        parts = video_file.stem.split('_')
        if len(parts) >= 4:
            try:
                file_month = int(parts[1])  # MM
                file_day = int(parts[2])    # DD
                
                # Compare dates (month first, then day)
                if (file_month > after_month) or (file_month == after_month and file_day > after_day):
                    filtered.append(video_file)
                    
            except (ValueError, IndexError):
                # If filename doesn't match pattern, include it to be safe
                filtered.append(video_file)
    
    print(f"Files AFTER {after_month:02d}/{after_day:02d}: {len(filtered)} files to process")
    return filtered


def print_motion_based_summary(summary, total_videos):
    """Print motion-based processing summary"""
    print(f"\n" + "="*80)
    print("MOTION-REQUIRED BATCH FLY DETECTION SUMMARY")
    print("="*80)
    print(f"Total videos found: {total_videos}")
    
    # ADDED: Print tracking info
    if 'tracking_summary' in summary:
        tracking = summary['tracking_summary']
        print(f"Videos attempted: {tracking['videos_attempted']}")
        print(f"Videos completed: {tracking['videos_completed']}")
        print(f"Videos with errors: {tracking['videos_with_errors']}")
        print(f"Completion rate: {tracking['completion_rate']}")
    
    print(f"Successful: {summary['successful_videos']}")
    print(f"Failed: {summary['failed_videos']}")
    print(f"Total time: {summary['total_processing_time']:.1f} seconds")
    print(f"Total MOVING flies detected: {summary['total_moving_flies']}")
    print(f"Total movement confirmations: {summary['total_movement_detections']}")
    print(f"Frames saved: {summary['total_frames_saved']}")
    print(f"Average moving flies per video: {summary['average_flies_per_video']:.1f}")
    print(f"Videos with moving flies: {summary['videos_with_moving_flies']}/{summary['successful_videos']}")
    print(f"Average processing speed: {summary['average_fps']:.1f} fps")
    print("NOTE: Static debris and dirt spots are now filtered out!")
    print(f"Check 'video_tracking_list.csv' for detailed processing status of each video")
    print("="*80)

if __name__ == "__main__":
    print("MOTION-REQUIRED FLY DETECTION")  
    # GPU check
    try:
        if verify_gpu():
            print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"GPU not available: {e}")
    
    # Configuration
    folder_path = "/home/pc/LG/StickySonic_PSU_flies/yolo_dino/src/video_data/april_middle"
    
    # CHANGED: Create output directory name based on INPUT folder name
    input_folder_name = Path(folder_path).name
    today = datetime.now().strftime("%m%d%y")
    output_directory = f"/home/pc/LG/StickySonic_PSU_flies/PSU_detectioncheck_data/{today}_{input_folder_name}_detection"
    
    print(f"Input folder: {input_folder_name}")
    print(f"Output directory: {output_directory}")
    
    if not Path(folder_path).exists():
        print(f"Folder not found: {folder_path}")
        print("Please update the folder_path variable")
        results, summary = [], {'error': 'Folder not found'}
    else:
        results, summary = process_folder_motion_based(
            folder_path=folder_path,
            output_directory=output_directory
        )
    
    if results:
        print(f"\nMotion-required batch processing complete!")
        print(f"Results saved to: {output_directory}")
    else:
        print(f"\nBatch processing failed or no videos found")
        if 'error' in summary:
            print(f"Error: {summary['error']}")
        sys.exit(1)