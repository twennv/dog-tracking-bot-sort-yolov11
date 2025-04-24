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

# --- AJOUT DU CHEMIN VERS BoT-SORT ---
BOT_SORT_PATH = os.path.join(os.getcwd(), "BoT-SORT")
sys.path.append(BOT_SORT_PATH)
sys.path.append(os.path.join(BOT_SORT_PATH, "tracker"))

from tracker.mc_bot_sort import BoTSORT

# --- CHEMINS ---
WEIGHTS_PATH = "./yolo11m.pt"
VIDEO_PATH = "./videos/Vidéo_test.mp4"
OUTPUT_PATH = "./output/output_video.mp4"

# --- CONFIGURATION DU PÉRIPHÉRIQUE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# --- DÉFINITION DES ARGUMENTS ---
class Args:
    source = VIDEO_PATH  # Chemin de la vidéo d'entrée
    weights = WEIGHTS_PATH  # Poids YOLOv11
    img_size = 960  # Taille de l'image
    conf_thres = 0.4  # Seuil de confiance
    iou_thres = 0.45  # Seuil IOU pour le NMS
    device = str(device)
    track_high_thresh = 0.4  # Seuil de confiance pour le tracking
    track_low_thresh = 0.10  # Seuil bas de détection
    track_buffer = 100  # Nombre de frames pour conserver un track perdu
    track_thresh = 50
    match_thresh = 0.9  # Seuil de correspondance pour le tracking
    save_path = OUTPUT_PATH  # Vidéo de sortie
    new_track_thresh = 0.8  # Seuil pour initialiser un nouveau track
    proximity_thresh = 0.6  # Seuil de proximité pour le ReID
    appearance_thresh = 0.4
    with_reid = False
    #fast_reid_config = "fast_reid/configs/Market1501/sbs_R50.yml"  # Fichier de config du modèle ReID
    #fast_reid_weights = "converted_reid.pth"  # Poids du modèle ReID
    fast_reid_config = "fast_reid/configs/MOT17/sbs_S50.yml"
    fast_reid_weights = "fast_reid/pretrained_models/mot17_sbs_S50.pth"
    cmc_method = "None"  # Méthode de compensation du mouvement caméra [None, orb, ecc]
    name = "BoT-SORT"
    ablation = False
    mot20 = False
    fp16 = False  # Half precision
    min_box_area = 50

args = Args()

# --- INITIALISATION DU MODÈLE ET DU TRACKER ---
print(f"Chargement des poids depuis {WEIGHTS_PATH}...")
model = YOLO(args.weights)
print("Modèle correctement chargé")

tracker = BoTSORT(args)

np.float = float  # Compatibilité

# --- PRÉPARATION DE LA VIDÉO ---
cap = cv2.VideoCapture(args.source)
if not cap.isOpened():
    print(f"Erreur: Impossible d'ouvrir la vidéo à {args.source}")
    sys.exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height))

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

        # Détection avec YOLOv8
        results = model(frame, classes=[16], verbose=False)[0]

        detections = []
        for det in results.boxes:
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            conf = det.conf.item()
            detections.append([x1, y1, x2, y2, conf, 16])  # Classe 16 = dog

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

            # Trouver la bbox de détection la plus proche via IoU
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1), (x1 + tw + 4, y1 + th + baseline + 4), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        cv2.imshow("BoT-SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Erreur majeure: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
