from kalman import KalmanFilter
from depth.depth import get_box_centers_with_disparity
import math
import numpy as np
import scipy.optimize.linear_sum_assignment as linear_assignment
FRAMES = 10
MAX_MISSES = 5
trackers = []


# cost matrix base on Euclidean distance to compare the center of detections and predictions
def calculate_cost_matrix(detections, predictions):

    N = len(detections)
    M = len(predictions)

    matrix = np.zeros((N, M))

    for i in N:
        det = detections[i]
        for j in M:
            pred = predictions[j]

            dist = math.sqrt((det[0]-pred[0])**2 + (det[1]-pred[1])**2 + (det[2]-pred[2])**2)
            matrix[i, j] = dist
    
    return matrix

for i in range(FRAMES): # will have to actually loop over video frames

    detections = []
    predictions = []

    # prediction step for all trackers
    for kf in trackers:
        predictions.append(kf.predict())

    objects = get_box_centers_with_disparity()  # hypothetical function to detect objects
    detections = [obj.position for obj in objects]

    # calculate cost matrix and perform association between detections and predictions
    cost_matrix = calculate_cost_matrix(detections, predictions)
    matched_pairs, unmatched_tracks, unmatched_detections = linear_assignment(cost_matrix)  # hungarian algorithm

    # update kalman filters for matched pairs
    for track_idx, detection_idx in matched_pairs:
        trackers[track_idx].update(detections[detection_idx])
        trackers[track_idx].undetected_count = 0

    # handle unmatched tracks
    for track_idx in unmatched_tracks:

        # increment undetected count but still keep the tracker (object might be occluded temporarily)
        trackers[track_idx].undetected_count += 1
        # if object does not reappear for a certain number of frames, remove the tracker (object lost)
        if trackers[track_idx].undetected_count > MAX_MISSES:
            del trackers[track_idx]
    
    # create new trackers for unmatched detections (probably new objects)
    for detection_idx in unmatched_detections:
        kf = KalmanFilter()
        kf.update(detections[detection_idx])
        trackers.append(kf)


    
    