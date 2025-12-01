import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(PROJECT_ROOT)

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
    
    return results

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


def get_box_centers_with_disparity(img_left, img_right, model):
    """
    Computes the disparity map from a stereo pair and returns a list of tuples (cx, cy, disparity)
    for each bounding box detected by the YOLO model.

    Args:
        img_left  (np.ndarray): Left stereo image (BGR format).
        img_right (np.ndarray): Right stereo image (BGR format).
        results   (ultralytics.engine.results.Results): YOLO detection results.

    Returns:
        list of (int, int, float): A list of tuples where each tuple contains:
                                   (center_x, center_y, disparity_value).
    """
    
    results=predict(img_left,model)
    r = results[0]
    # --- 1. Convert both images to grayscale for StereoBM ---
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # --- 2. Configure the StereoBM block matcher ---
    min_disp = 0
    num_disp = 5 * 16        # must be divisible by 16
    block_size = 15

    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    stereo.setMinDisparity(min_disp)
    stereo.setDisp12MaxDiff(200)
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleRange(10)
    stereo.setSpeckleWindowSize(1)

    # --- 3. Compute disparity map ---
    disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # --- 4. Extract center coordinates and disparity values ---
    results_list = []   # List of tuples: (cx, cy, disparity)
    f = 700.0        # fokalna dužina u pikselima
    B = 0.10         # baseline u metrima (10 cm)
    f_x = f          # Focal length x
    f_y = f          # Focal length y
    c_x = img_left.shape[1] / 2 # Principal point x (approx)
    c_y = img_left.shape[0] / 2 # Principal point y (approx)

    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()

        # Compute center of bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # NumPy image indexing uses [row, column] = [y, x]
        disparity_value = disp[cy, cx]
        Z = (f * B) / disparity_value
        # Inverse Projection (Pixels -> Camera Coordinates)
        X = Z * (cx - c_x) / f_x         # X in meters
        Y = Z * (cy - c_y) / f_y         # Y in meters

        if not np.isfinite(X) or not np.isfinite(Y) or not np.isfinite(Z) or Z <= 0 or Z > 15.0:
            continue

        results_list.append(( X, Y, Z))

    return results_list, r
