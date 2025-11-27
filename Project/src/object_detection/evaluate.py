from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from detection_inference import *


CLASSES_OF_INTEREST = ['car', 'pedestrian', 'cyclist',]


#----- METHODS TO READ THE LABELS.TXT FILE -----#

def load_labels(label_path):
    """
    Reads labels.txt and returns a dict:
        { frame_id (int): [obj1, obj2, ...] }

    Each obj is a dict with parsed fields.
    :param label_path: Path to labels.txt file
    :return: Dict mapping frame indices to lists of label dicts
    """
    
    # Initialize dictionary to hold labels by frame
    labels_by_frame = defaultdict(list)
    
    # Ensure label_path is a Path object
    label_path = Path(label_path)
    
    with label_path.open("r") as file:
        for line in file:
            # Remove whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue
            
            # The labels format is the following:
            # <frame_idx> <object_id> <class_label> <truncated> <occluded> <alpha> BBOX (x1, y1, x2, y2) 
            parts = line.split()
            
            frame_idx = int(parts[0])
            object_idx = int(parts[1])
            class_label = parts[2]
            bbox = list(map(float, parts[6:10]))  # x1, y1, x2, y2
            
            obj = {
                "frame_idx": frame_idx,
                "object_idx": object_idx,
                "class_label": class_label,
                "bbox": bbox,
            }
            labels_by_frame[frame_idx].append(obj)
    
    return labels_by_frame

def plot_frame(labels, image):
    """
    Plot ground truth labels on the image.
    :param labels: List of label dicts for the frame
    :param image: Original image
    """
    
    image = image.copy()
    
    for obj in labels:
        bbox = obj['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        class_label = obj['class_label']
        object_idx = obj['object_idx']
        
        label = f"{class_label}"
        cv2.rectangle(
            image,
            pt1=(x1, y1), pt2=(x2, y2),
            color=(255, 0, 0), thickness=2,
        )
        cv2.putText(
            image, f"{label} ID:{object_idx}",
            org=(x1, max(15, y1 - 10)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(255, 0, 0), thickness=2,
        )
    
    cv2.imshow(f"Ground Truth Labels", image)
    # display every 50 ms
    cv2.waitKey(50)
       
def plot_sequence(labels_by_frame, video_path):
    """
    Plot ground truth labels for a sequence of images.
    :param labels_by_frame: Dict mapping frame indices to lists of label dicts
    :param video_path: Path to the directory containing image frames
    """
    image_files = sorted(video_path.glob("*.png"))
    
    for frame_idx, image_file in enumerate(image_files):
        image = cv2.imread(str(image_file))
        labels = labels_by_frame.get(frame_idx, [])
        plot_frame(labels, image)
      
        
#----- EVALUATION METHODS    -----#

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    IoU = Area of Overlap / Area of Union
    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    """
    # Determine the coordinates of the intersection box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    interection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection_area = interection_width * intersection_height

    if intersection_area == 0:
        return 0.0
    
    # Compute the area of both bounding boxes
    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    # Compute the area of union
    union = area1 + area2 - intersection_area
    if union <= 0:
        return 0.0

    return intersection_area / union


def get_image_files(sequence_path):
    """
    Get sorted list of image files in the sequence directory.
    :param sequence_path: Path to the sequence directory
    :return sorted list of image paths in the sequence.
    """
    sequence_path = Path(sequence_path)
    image_files = sorted(sequence_path.glob("*.png"))
    print(f"Found {len(image_files)} images in {sequence_path}")
    return image_files


def get_gt_for_frame(labels_by_frame, frame_idx):
    """
    Get ground truth boxes and class labels for a specific frame.
    :param labels_by_frame: Dict mapping frame indices to lists of label dicts
    :param frame_idx: Index of the frame
    :return: (gt_boxes, gt_classes)
    """
    gt_objects = labels_by_frame.get(frame_idx, [])
    gt_boxes = [obj["bbox"] for obj in gt_objects]
    gt_classes = [obj["class_label"].lower() for obj in gt_objects]
    return gt_boxes, gt_classes


def run_model_on_image(model, image, confidence_threshold=0.6, device="cpu"):
    """
    Run the YOLO model on a single image and return sorted predictions
    :param model: Loaded YOLO model
    :param image: Input image (numpy array)
    :param confidence_threshold: Confidence threshold for detections
    :param device: Device to run the model on ('cpu' or 'cuda')
    :return: 
        prediction_xyxy: (N, 4) numpy array of predicted boxes
        prediction_confs: (N,) numpy array of confidence scores
        prediction_class: (N,) numpy array of class indices
    """
    # Run model inference
    results = model.predict(
        source=[image],
        conf=confidence_threshold,
        imgsz=640,
        device=device,
        verbose=False,
    )[0]

    boxes = results.boxes
    # If no boxes detected then return None
    if boxes is None or len(boxes) == 0:
        return None, None, None

    # Extract predictions
    prediction_xyxy = boxes.xyxy.cpu().numpy() 
    prediction_confs = boxes.conf.cpu().numpy() 
    prediction_class = boxes.cls.cpu().numpy().astype(int) 

    # Sort predictions by confidence (descending)
    order = np.argsort(-prediction_confs)
    prediction_xyxy = prediction_xyxy[order]
    prediction_confs = prediction_confs[order]
    prediction_class = prediction_class[order]

    return prediction_xyxy, prediction_confs, prediction_class

def match_predictions_to_gt(
    prediction_xyxy, prediction_confs, prediction_class,
    gt_boxes, gt_classes, idx_to_name, iou_threshold=0.5,):
    """
    Match predictions to ground truth using IoU and class labels.
    :param prediction_xyxy: (N, 4) numpy array of predicted boxes
    :param prediction_confs: (N,) numpy array of confidence scores
    :param prediction_class: (N,) numpy
    :param gt_boxes: list of GT boxes [[x1, y1, x2, y2], ...]
    :param gt_classes: list of GT class names
    :param idx_to_name: dict or list mapping class indices to class names
    :param iou_threshold: IoU threshold to consider a match
    :return:
      matched: list of (predicted_class_name, is_tp)
      unmatched_gt_classes: list of GT class names that were not matched (FN)
    """
    # list of tuples (pred_class_name, is_tp)
    matched = []  
    gt_matched = [False] * len(gt_boxes)

    # No predictions: all GT are FN
    if prediction_xyxy is None:
        return [], gt_classes[:]  # all GT unmatched

    for predicted_box, class_idx, confidence in zip(prediction_xyxy, prediction_class, prediction_confs):
        predicted_class_name = str(idx_to_name[class_idx]).lower()

        # Skip classes not of interest
        if predicted_class_name.lower() not in CLASSES_OF_INTEREST:
            continue

        best_iou = 0.0
        best_gt_idx = -1

        # Find best matching GT of the same class
        for j, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
            if gt_matched[j]:
                continue
            if gt_class != predicted_class_name:
                continue

            iou = compute_iou(predicted_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # If best match exceeds IoU threshold mark as TP
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            matched.append((predicted_class_name, True))   # TP
        # Otherwise FP
        else:
            matched.append((predicted_class_name, False))  # FP

    # Any GT that are not matched will be marked as FN
    unmatched_gt_classes = [
        gt_class for matched_flag, gt_class in zip(gt_matched, gt_classes) if not matched_flag
    ]

    return matched, unmatched_gt_classes


def update_stats(stats, overall, matched, unmatched_gt_classes):
    """
    Update stats dicts given matched predictions and unmatched GT classes.
    :param matched: list of (cls_name, is_tp)
    :param unmatched_gt_classes: list of cls_name
    :param stats: per-class stats dict
    :param overall: overall stats dict
    :return: updated stats and overall dicts
    """
    for cls_name, is_tp in matched:
        if is_tp:
            stats[cls_name]["tp"] += 1
            overall["tp"] += 1
        else:
            stats[cls_name]["fp"] += 1
            overall["fp"] += 1

    for gt_class in unmatched_gt_classes:
        stats[gt_class]["fn"] += 1
        overall["fn"] += 1
        
    return stats, overall


def compute_and_print_metrics(stats, overall, iou_threshold, confidence_threshold, save_path=None):
    """
    Compute precision / recall / F1 from stats and print them.
    :param stats: per-class stats dict
    :param overall: overall stats dict
    :param iou_threshold: IoU threshold used
    :param confidence_threshold: Confidence threshold used
    :param save_path: Optional path to save the metrics
    :return: None
    """
    print("\n=== Per-class metrics (IoU >= {:.2f}, conf >= {:.2f}) ===".format(
        iou_threshold, confidence_threshold))

    for cls, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"\nClass: {cls}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1-score:  {f1:.3f}")

    tp, fp, fn = overall["tp"], overall["fp"], overall["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print("\n=== Overall metrics ===")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    
    if save_path is not None:
        # Save the metrics to a csv file
        save_path = Path(save_path)
        with save_path.open("w") as f:
            f.write("Class,TP,FP,FN,Precision,Recall,F1-score\n")
            for cls, s in stats.items():
                tp, fp, fn = s["tp"], s["fp"], s["fn"]
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f.write(f"{cls},{tp},{fp},{fn},{prec:.3f},{rec:.3f},{f1:.3f}\n")
            f.write(f"Overall,{tp},{fp},{fn},{prec:.3f},{rec:.3f},{f1:.3f}\n")
        print(f"\nMetrics saved to {save_path}")
        


def evaluate_model_on_sequence(
    sequence_path, labels_by_frame, model,
    confidence_threshold=0.6, iou_threshold=0.5, device='cpu'):
    """
    Run evaluation of the model on a sequence of images.
    :param sequence_path: Path to the sequence directory
    :param labels_by_frame: Dict mapping frame indices to lists of label dicts
    :param model: Loaded YOLO model
    :param confidence_threshold: Confidence threshold for detections
    :param iou_threshold: IoU threshold to consider a match
    :param device: Device to run the model on ('cpu' or 'cuda')
    :return: None
    """
    # Prepare the image files
    image_files = get_image_files(sequence_path)

    # Initialize statistics for the evaluation
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    overall = {"tp": 0, "fp": 0, "fn": 0}
    
    # Dictionary mapping class indices to class names
    idx_to_name = model.names
    print(f"Model class names: {idx_to_name}")

    # Loop over frames
    for frame_idx, image_file in enumerate(tqdm(image_files)):
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Warning: could not read {image_file}")
            continue

        # Ground truth for this frame
        gt_boxes, gt_classes = get_gt_for_frame(labels_by_frame, frame_idx)

        # Run model
        prediction_xyxy, prediction_confs, prediction_class = run_model_on_image(
            model, image, confidence_threshold, device
        )

        # Match predictions ↔ GT
        matched, unmatched_gt_classes = match_predictions_to_gt(
            prediction_xyxy,
            prediction_confs,
            prediction_class,
            gt_boxes,
            gt_classes,
            idx_to_name,
            iou_threshold=iou_threshold,
        )

        # Update global stats
        stats, overall = update_stats(stats, overall, matched, unmatched_gt_classes)

    # Final metrics
    compute_and_print_metrics(stats, overall, iou_threshold, confidence_threshold)


if __name__ == '__main__':
    
    sequence_1_path = PROJECT_ROOT / 'data/34759_final_project_rect/seq_01/'        # image 02
    sequence_2_path = PROJECT_ROOT / 'data/34759_final_project_rect/seq_02/'        # image 02
    
    # Sequences 1 and 2 provide a validation set for the trained model
    # Get all GT labels for frame i
    labels_by_frame = load_labels(sequence_2_path / 'labels.txt')
   
    plot_sequence(labels_by_frame, sequence_2_path / 'image_02' / 'data')
   
    # model = load_model(PROJECT_ROOT / 'working_files/weights/best.pt')
    
    # evaluate_model_on_sequence(
    #     sequence_2_path / 'image_02' / 'data',
    #     labels_by_frame,
    #     model,
    #     confidence_threshold=0.5,
    #     iou_threshold=0.5,
    #     device='cpu',
    # )