
from scipy.optimize import linear_sum_assignment as linear_assignment
from pathlib import Path

import math
import numpy as np
import cv2

from .kalman import KalmanFilter
from ..depth.depth import get_box_centers_with_disparity


def is_near_border(tracker_position, img_width, img_height):

    x = tracker_position[0]
    y = tracker_position[1]
    if x < BORDER_WIDTH_THRESHOLD or y < BORDER_HEIGHT_THRESHOLD or (img_width - x) < BORDER_WIDTH_THRESHOLD or (img_height - y) < BORDER_HEIGHT_THRESHOLD:
        return True
    return False



def calculate_cost_matrix(detections, predictions):

    N = len(detections)
    M = len(predictions)

    matrix = np.zeros((N, M))

    for i in range(N):
        det = detections[i]
        for j in range(M):
            pred = predictions[j]

            dist = math.sqrt((det[0]-pred[0])**2 + (det[1]-pred[1])**2 + (det[2]-pred[2])**2)
            matrix[i, j] = dist
    
    return matrix



def hungarian_algorithm(cost_matrix, MAX_DIST=20.0):

    row_ind, col_ind = linear_assignment(cost_matrix)
    
    matched_pairs = []
    
    for r, c in zip(row_ind, col_ind):
        # Only accept a match if the cost is below the defined threshold
        if cost_matrix[r, c] < MAX_DIST: 
            matched_pairs.append((r, c))
             
    matched_rows = {r for r, c in matched_pairs}
    matched_cols = {c for r, c in matched_pairs}

    # Recalculate unmatched_detections
    num_detections = cost_matrix.shape[0]
    unmatched_detections = sorted(list(set(range(num_detections)) - matched_rows))

    # Recalculate unmatched_tracks
    num_tracks = cost_matrix.shape[1]
    unmatched_tracks = sorted(list(set(range(num_tracks)) - matched_cols))

    return matched_pairs, unmatched_tracks, unmatched_detections



def plot_tracking_results(image, detections, trackers):

    image = image.copy()

    # results can be a single result or an iterable of results.
    results_iter = detections if isinstance(detections, (list, tuple)) else [detections]

    for result in results_iter:
        boxes = result.boxes  # Boxes object

        # Convert tensors to numpy arrays
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None  # (N, 4)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None   # (N,)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else None     # (N,)

        if xyxy is None or len(xyxy) == 0:
            continue

        # Iterate through detections
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box)
            conf = float(confs[i]) if confs is not None else 0.0
            cls_id = int(clss[i]) if clss is not None else -1

            label = f"{cls_id}: {conf:.2f}"
            cv2.rectangle(
                image,
                pt1=(x1, y1), pt2=(x2, y2),
                color=(0, 255, 0), thickness=2,
            )
            cv2.putText(
                image, label,
                org=(x1, max(15, y1 - 10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 255, 0), thickness=2,
            )

    for kf in trackers:
        print ("Tracker state:", kf[0], kf[1])
        x, y = int(kf[0]), int(kf[1])
        cv2.circle(
            image,
            center=(x, y),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )

    cv2.imshow(f"Tracking Result", image)

    cv2.waitKey(50)



def tracking_loop():

    image_left_files = sorted(video_left_path.glob("*.png"))
    image_right_files = sorted(video_right_path.glob("*.png"))
    print(f"Found {len(image_left_files)} images in {video_left_path}")
    print(f"Found {len(image_right_files)} images in {video_right_path}")

    for image_left, image_right in zip(image_left_files, image_right_files):
        frame_left = cv2.imread(str(image_left))
        frame_right = cv2.imread(str(image_right))

        detections = []
        predictions = []
        trackers_to_delete = []

        # prediction step for all trackers
        for kf in trackers:
            predictions.append(kf.predict())
        

        objects, boxes_to_display = get_box_centers_with_disparity(frame_left, frame_right)  # hypothetical function to detect objects
        detections = [obj for obj in objects]
        
        plot_tracking_results(frame_left, boxes_to_display, predictions)

        # calculate cost matrix and perform association between detections and predictions
        cost_matrix = calculate_cost_matrix(detections, predictions)
        matched_pairs, unmatched_tracks, unmatched_detections = hungarian_algorithm(cost_matrix)  # hungarian algorithm

        # update kalman filters for matched pairs
        for detection_idx, track_idx in matched_pairs:
            trackers[track_idx].update(detections[detection_idx])
            trackers[track_idx].undetected_count = 0

        # handle unmatched tracks
        for track_idx in unmatched_tracks:
            # increment undetected count but still keep the tracker (object might be occluded temporarily)
            if track_idx < len(trackers):
                trackers[track_idx].undetected_count += 1
                
                # Check for removal condition
                if trackers[track_idx].undetected_count > MAX_MISSES or is_near_border(trackers[track_idx].x[0:2][0], frame_left.shape[1], frame_left.shape[0]):
                    trackers_to_delete.append(track_idx)

        for track_idx in sorted(trackers_to_delete, reverse=True):
            del trackers[track_idx]
            

        # create new trackers for unmatched detections (probably new objects)
        for detection_idx in unmatched_detections:
            kf = KalmanFilter()
            kf.update(detections[detection_idx])
            trackers.append(kf)


    cv2.destroyAllWindows()
        

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    print(PROJECT_ROOT)

    video_left_path = PROJECT_ROOT / 'datasets' / '34759_final_project_rect' / 'seq_02' / 'image_02' / 'data'
    video_right_path = PROJECT_ROOT / 'datasets' / '34759_final_project_rect' / 'seq_02' / 'image_03' / 'data'

    MAX_MISSES = 50 # maximum number of consecutive misses before deleting a tracker
    BORDER_WIDTH_THRESHOLD = 300 # pixels
    BORDER_HEIGHT_THRESHOLD = 100 # pixels

    trackers = []

    tracking_loop()