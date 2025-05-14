# --- IMPORTS ---
#import argparse
import os
import sys
import cv2
import torch
import numpy as np
#import time
#from pathlib import Path
from ultralytics import YOLO
from shapely.geometry import box

from tracker.mc_bot_sort import BoTSORT

# --- ADD PATH TO BoT-SORT ---
BOT_SORT_PATH = os.path.join(os.getcwd(), "BoT-SORT")
sys.path.append(BOT_SORT_PATH)
sys.path.append(os.path.join(BOT_SORT_PATH, "tracker"))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fix DLL error

# --- PATHS ---
WEIGHTS_PATH = "./yolo11m.pt"
VIDEO_PATH = "./videos/IMAG0670.MOV"
OUTPUT_PATH = "./output/output_video.mp4"

# --- DEVICE CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# --- DEFINE ARGUMENTS ---
class Args:
    source = VIDEO_PATH  # Input video path
    weight = WEIGHTS_PATH  # YOLOv11 weight
    img_size = 960  # Image size
    conf_thres = 0.4  # Confidence threshold
    iou_thres = 0.45  # IOU threshold for NMS
    device = str(device)
    track_high_thresh = 0.4  # High confidence threshold for tracking
    track_low_thresh = 0.10  # Low detection threshold
    track_buffer = 180  # Frames to keep a lost track
    match_thresh = 0.8  # Matching threshold for tracking
    save_path = OUTPUT_PATH  # Output video path
    new_track_thresh = 0.6  # Threshold for new track creation
    proximity_thresh = 0.7  # Proximity threshold for ReID
    appearance_thresh = 0.4
    with_reid = False
    fast_reid_config = "./re-id/config.yaml"
    fast_reid_weight = "./re-id/BIFOR_nan.pth"
    cmc_method = "None"  # Camera motion compensation method [None, orb, ecc]
    name = "BoT-SORT"
    ablation = False
    mot20 = False
    fp16 = False  # Half precision
    min_box_area = 30

args = Args()

# --- MODEL AND TRACKER INITIALIZATION ---
print(f"Chargement des poids depuis {WEIGHTS_PATH}...")
model = YOLO(args.weights)
print("Modèle correctement chargé")

tracker = BoTSORT(args)

np.float = float  # Compatibilité

# --- VIDEO PREPARATION ---
cap = cv2.VideoCapture(args.source)
if not cap.isOpened():
    print(f"Erreur: Impossible d'ouvrir la vidéo à {args.source}")
    sys.exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height))

# --- Dynamic display adaptation ---
scale_factor = (width * height) / (1280 * 720)

font_scale = max(0.5, min(2.0, scale_factor ** 0.5 * 0.5))
bbox_thickness = max(2, int(font_scale * 5))
text_thickness = max(1, round(font_scale * 2))
display_scale = min(1.0, max(0.3, 720 / height))

def get_color_from_id(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

def compute_iou(bbox1, bbox2):
    """Calcule l'IoU entre deux boîtes [x1, y1, x2, y2]."""
    box1 = box(*bbox1)
    box2 = box(*bbox2)
    inter = box1.intersection(box2).area
    union = box1.union(box2).area
    return inter / union if union > 0 else 0

track_confidences = {}

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[16], verbose=False)[0]

        detections = []
        for det in results.boxes:
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            conf = det.conf.item()
            detections.append([x1, y1, x2, y2, conf, 16])  # Class 16 = dog

        detections = np.array(detections)
        tracked_objects = tracker.update(detections, frame)

        for obj in tracked_objects:
            if hasattr(obj, 'tlbr'):
                bbox = obj.tlbr
                track_id = obj.track_id
            else:
                bbox = obj[:4]
                track_id = int(obj[4])

            x1, y1, x2, y2 = map(int, bbox)
            color = get_color_from_id(track_id)

            best_iou = 0
            best_conf = None
            for det in detections:
                iou = compute_iou(det[:4], bbox)
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_conf = det[4]

            if best_conf is not None:
                track_confidences[track_id] = best_conf

            confidence = track_confidences.get(track_id, None)
            text = f"ID: {track_id} - {confidence:.2f}" if confidence is not None else f"ID: {track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thickness)
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            cv2.rectangle(frame, (x1, y1), (x1 + tw + 6, y1 + th + baseline + 6), color, -1)
            cv2.putText(frame, text, (x1 + 3, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)

        out.write(frame)

        display_frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow("BoT-SORT Tracking", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Critical error: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
