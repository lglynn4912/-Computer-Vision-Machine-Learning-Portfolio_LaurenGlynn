import math
import numpy as np
from collections import defaultdict

class FlyEventCounter:
    def __init__(self, spatial_threshold=50, temporal_threshold=15, min_event_duration=2):
        """
        Initialize fly event counter
        
        Args:
            spatial_threshold: Max distance (pixels) for same fly
            temporal_threshold: Max frame gap for same fly
            min_event_duration: Minimum frames for valid event
        """
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.min_event_duration = min_event_duration
        
    def count_unique_fly_events(self, frame_detections):
        """
        Convert frame-by-frame detections into unique fly events
        
        Args:
            frame_detections: List of detections per frame
                [{'frame': 1, 'detections': [{'bbox': (x1,y1,x2,y2), 'confidence': 0.8}]}, ...]
        
        Returns:
            List of unique fly events
        """
        # Step 1: Flatten all detections with frame info
        all_detections = []
        for frame_data in frame_detections:
            frame_num = frame_data['frame']
            for detection in frame_data['detections']:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                all_detections.append({
                    'frame': frame_num,
                    'center': (center_x, center_y),
                    'bbox': bbox,
                    'confidence': detection['confidence'],
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                })
        
        if not all_detections:
            return []
        
        # Step 2: Sort by frame number
        all_detections.sort(key=lambda x: x['frame'])
        
        # Step 3: Group into events using trajectory tracking
        events = self._track_fly_trajectories(all_detections)
        
        # Step 4: Merge nearby events that might be the same fly
        merged_events = self._merge_nearby_events(events)
        
        # Step 5: Filter out events that are too short
        valid_events = [e for e in merged_events if e['duration_frames'] >= self.min_event_duration]
        
        return valid_events
    
    def _track_fly_trajectories(self, detections):
        """Track fly trajectories to group detections into events"""
        events = []
        used_detection_indices = set()
        
        for i, detection in enumerate(detections):
            if i in used_detection_indices:
                continue
                
            # Start new trajectory
            trajectory = [i]
            used_detection_indices.add(i)
            
            # Look for continuation of this trajectory
            current_frame = detection['frame']
            current_center = detection['center']
            
            # Search forward in time for connected detections
            for j in range(i + 1, len(detections)):
                if j in used_detection_indices:
                    continue
                    
                candidate = detections[j]
                frame_gap = candidate['frame'] - current_frame
                
                # Skip if too far in time
                if frame_gap > self.temporal_threshold:
                    break
                    
                # Check spatial distance
                distance = self._calculate_distance(current_center, candidate['center'])
                
                # If close enough, add to trajectory
                if distance <= self.spatial_threshold:
                    trajectory.append(j)
                    used_detection_indices.add(j)
                    current_frame = candidate['frame']
                    current_center = candidate['center']
            
            # Create event from trajectory
            if trajectory:
                event = self._create_event_from_trajectory(trajectory, detections)
                events.append(event)
        
        return events
    
    def _create_event_from_trajectory(self, trajectory_indices, detections):
        """Create event object from trajectory indices"""
        trajectory_detections = [detections[i] for i in trajectory_indices]
        
        frames = [d['frame'] for d in trajectory_detections]
        centers = [d['center'] for d in trajectory_detections]
        confidences = [d['confidence'] for d in trajectory_detections]
        
        # Calculate trajectory stats
        start_frame = min(frames)
        end_frame = max(frames)
        duration = end_frame - start_frame + 1
        
        # Calculate movement distance
        total_distance = 0
        for i in range(1, len(centers)):
            total_distance += self._calculate_distance(centers[i-1], centers[i])
        
        # Find best representative detection (highest confidence)
        best_detection_idx = max(range(len(trajectory_detections)), 
                               key=lambda i: trajectory_detections[i]['confidence'])
        representative_detection = trajectory_detections[best_detection_idx]
        
        return {
            'event_id': None,  # Will be assigned later
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration_frames': duration,
            'num_detections': len(trajectory_detections),
            'total_movement': total_distance,
            'avg_confidence': np.mean(confidences),
            'max_confidence': max(confidences),
            'representative_detection': representative_detection,
            'trajectory_centers': centers,
            'all_detections': trajectory_detections
        }
    
    def _merge_nearby_events(self, events):
        """Merge events that are likely the same fly separated by brief gaps"""
        if len(events) <= 1:
            return events
            
        # Sort events by start frame
        events.sort(key=lambda x: x['start_frame'])
        
        merged_events = []
        current_event = events[0]
        
        for next_event in events[1:]:
            # Check if events should be merged
            time_gap = next_event['start_frame'] - current_event['end_frame']
            
            if time_gap <= self.temporal_threshold:
                # Check spatial proximity between end of current and start of next
                current_end_center = current_event['trajectory_centers'][-1]
                next_start_center = next_event['trajectory_centers'][0]
                distance = self._calculate_distance(current_end_center, next_start_center)
                
                if distance <= self.spatial_threshold * 1.5:  # Slightly more lenient for merging
                    # Merge events
                    current_event = self._merge_two_events(current_event, next_event)
                    continue
            
            # Can't merge, save current and move to next
            merged_events.append(current_event)
            current_event = next_event
        
        # Add the last event
        merged_events.append(current_event)
        
        # Assign event IDs
        for i, event in enumerate(merged_events):
            event['event_id'] = i + 1
        
        return merged_events
    
    def _merge_two_events(self, event1, event2):
        """Merge two events into one"""
        return {
            'event_id': event1['event_id'],
            'start_frame': min(event1['start_frame'], event2['start_frame']),
            'end_frame': max(event1['end_frame'], event2['end_frame']),
            'duration_frames': max(event1['end_frame'], event2['end_frame']) - min(event1['start_frame'], event2['start_frame']) + 1,
            'num_detections': event1['num_detections'] + event2['num_detections'],
            'total_movement': event1['total_movement'] + event2['total_movement'],
            'avg_confidence': (event1['avg_confidence'] + event2['avg_confidence']) / 2,
            'max_confidence': max(event1['max_confidence'], event2['max_confidence']),
            'representative_detection': event1['representative_detection'] if event1['max_confidence'] > event2['max_confidence'] else event2['representative_detection'],
            'trajectory_centers': event1['trajectory_centers'] + event2['trajectory_centers'],
            'all_detections': event1['all_detections'] + event2['all_detections']
        }
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_event_summary(self, events):
        """Generate summary statistics for events"""
        if not events:
            return {
                'total_events': 0,
                'total_detections': 0,
                'avg_event_duration': 0,
                'avg_confidence': 0
            }
        
        total_detections = sum(e['num_detections'] for e in events)
        avg_duration = np.mean([e['duration_frames'] for e in events])
        avg_confidence = np.mean([e['avg_confidence'] for e in events])
        
        return {
            'total_events': len(events),
            'total_detections': total_detections,
            'avg_event_duration': avg_duration,
            'avg_confidence': avg_confidence,
            'events_by_duration': self._categorize_events_by_duration(events)
        }
    
    def _categorize_events_by_duration(self, events):
        """Categorize events by duration"""
        categories = {
            'brief (1-5 frames)': 0,
            'short (6-15 frames)': 0,
            'medium (16-30 frames)': 0,
            'long (31+ frames)': 0
        }
        
        for event in events:
            duration = event['duration_frames']
            if duration <= 5:
                categories['brief (1-5 frames)'] += 1
            elif duration <= 15:
                categories['short (6-15 frames)'] += 1
            elif duration <= 30:
                categories['medium (16-30 frames)'] += 1
            else:
                categories['long (31+ frames)'] += 1
        
        return categories

# Example usage for your finalize_count.py integration
def integrate_with_finalize_count(video_results):
    """
    Integrate event counting with your existing detection results
    
    Args:
        video_results: Results from your process_video function
    
    Returns:
        Enhanced results with unique event counts
    """
    counter = FlyEventCounter(
        spatial_threshold=50,    # Adjust based on your video resolution
        temporal_threshold=10,   # Max 10 frame gap between detections
        min_event_duration=2     # At least 2 frames for valid event
    )
    
    enhanced_results = []
    
    for video_result in video_results:
        video_detections = video_result.get('fly_detections', [])
        
        # Convert to format expected by event counter
        frame_detections = []
        for detection in video_detections:
            frame_detections.append({
                'frame': detection['frame'],
                'detections': detection['fly_detections']
            })
        
        # Count unique events
        unique_events = counter.count_unique_fly_events(frame_detections)
        event_summary = counter.get_event_summary(unique_events)
        
        # Add to results
        enhanced_result = video_result.copy()
        enhanced_result['unique_fly_events'] = unique_events
        enhanced_result['event_summary'] = event_summary
        enhanced_result['unique_fly_count'] = len(unique_events)
        
        enhanced_results.append(enhanced_result)
    
    return enhanced_results