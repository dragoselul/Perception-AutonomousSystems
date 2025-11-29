from ultralytics import YOLO
import cv2
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(PROJECT_ROOT)

CLASSES_OF_INTEREST = ['pedestrian', 'cyclist', 'car']


def load_model(model_path):
    """ 
    Load the YOLO model from the specified weights file.
    :return: Loaded YOLO model
    """
    try:
        model = YOLO(model_path)
        # Set to validation mode
        model.fuse()
        
        print("Model loaded from best.pt")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict(images, model, confidence_threshold=0.6, device='cpu'):
    """ 
    Perform inference on a list of images using the YOLO model.
    :param images: List of image file paths or numpy arrays
    :param model: Loaded YOLO model
    :param confidence_threshold: Confidence threshold for detections
    :return: Inference results
    """
    results = model.predict(
        source=images, 
        conf=confidence_threshold, 
        imgsz=640, device=device, verbose=False,
    )
    
    # Filter results to include only classes of interest
    results_filtered = []
    for result in results:
        boxes = result.boxes
        if boxes is not None and hasattr(boxes, "cls"):
            cls_ids = boxes.cls.cpu().numpy()
            mask = [i for i, cls_id in enumerate(cls_ids) if model.names[int(cls_id)].lower() in CLASSES_OF_INTEREST]
            if len(mask) > 0:
                filtered_boxes = boxes[mask]
                result.boxes = filtered_boxes
            else:
                result.boxes = type(boxes)()  # Empty boxes
        results_filtered.append(result)
    return results_filtered


def plot_predictions(results, image):
    """
    Plot the detection results on the image.
    :param result: Detection result (from YOLO model)
    :param image: Original image
    """
    
    image = image.copy()

    # results can be a single result or an iterable of results.
    results_iter = results if isinstance(results, (list, tuple)) else [results]

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
    cv2.imshow(f"Detection Result", image)
    # display every 50 ms
    cv2.waitKey(50)
    

def object_detection_video(video_path, model, confidence_threshold=0.6):
    """
    Perform object detection on a video file.
    :param video_path: Path to the images from the video
    :param model: Loaded YOLO model
    :param confidence_threshold: Confidence threshold for detections
    """
    image_files = sorted(video_path.glob("*.png"))
    print(f"Found {len(image_files)} images in {video_path}")
    
    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        
        results = predict([frame], model, confidence_threshold)
        plot_predictions(results[0], frame)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_model(PROJECT_ROOT / 'working_files/weights/best_finetune.pt')
    video_path = PROJECT_ROOT / 'data/34759_final_project_rect/seq_02/image_03/data'
    object_detection_video(video_path, model, confidence_threshold=0.6)
    