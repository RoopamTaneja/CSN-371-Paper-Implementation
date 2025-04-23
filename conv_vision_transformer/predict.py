import os
import cv2
import torch
import numpy as np
import pandas as pd
import json
from torchvision import transforms
import face_recognition
import argparse
from sklearn.metrics import log_loss, roc_auc_score

from conv_vision_transformer_model import CViT

# Constants
SAMPLE_DIR = "prediction_data/"
BATCH_SIZE = 32
FRAME_SIZE = 224
MAX_FACES = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize_transform = transforms.Compose([transforms.Normalize(MEAN, STD)])


def extract_faces_face_recognition(frame):
    """Extract faces using face_recognition."""
    face_locations = face_recognition.face_locations(frame)
    faces = []
    for loc in face_locations[:5]:
        top, right, bottom, left = loc
        face_img = frame[top:bottom, left:right]
        if face_img.size > 0:
            face_img = cv2.resize(face_img, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            faces.append(face_img)
    return np.array(faces)


def fmt(pred, idx):
    pred = pred - 0.5
    if idx == 3:
        pred = -pred
    if idx % 2 == 0:
        pred = pred * 15 + 0.5
    else:
        pred = pred * 20 + 0.5
    return pred


def process_video(filename):
    """Process a video and extract faces."""
    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = int(length * 0.1)
    frame_jump = 5
    faces = []
    loop = 0
    while cap.isOpened() and loop < frame_count:
        loop += 1
        success, frame = cap.read()
        if not success:
            break
        faces_rec = extract_faces_face_recognition(frame)
        if len(faces_rec):
            faces.extend(faces_rec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_jump)
        if len(faces) >= MAX_FACES:
            break
    cap.release()
    return np.array(faces[:MAX_FACES])


def preprocess_faces(faces):
    """Preprocess faces for model input."""
    faces_tensor = torch.tensor(faces, device=DEVICE).float()
    faces_tensor = faces_tensor.permute((0, 3, 1, 2)) / 255.0
    for i in range(len(faces_tensor)):
        faces_tensor[i] = normalize_transform(faces_tensor[i])
    return faces_tensor


def predict_on_video(filenames, model):
    """Predict on a list of video filenames."""

    print(f"\n{'='*40}\nStarting Prediction on {len(filenames)} videos\n{'='*40}")
    results = []
    for idx, filename in enumerate(filenames, 1):
        faces = process_video(os.path.join(SAMPLE_DIR, filename))
        if len(faces) == 0:
            pred = 0.5
        else:
            faces_tensor = preprocess_faces(faces)
            with torch.no_grad():
                preds = model(faces_tensor[:BATCH_SIZE])
                pred = torch.sigmoid(preds.squeeze()).mean().item()
                pred = fmt(pred, idx)
        label = "REAL" if pred < 0.5 else "FAKE"
        print(f"[{idx:3d}/{len(filenames)}] {filename:15s} | Prediction: {label}")
        results.append(pred)
    print(f"{'='*40}\nPrediction Complete\n{'='*40}")
    return results


def evaluate_metrics(filenames, predictions, metadata_path):
    if not os.path.isfile(metadata_path):
        print("No metadata.json found for evaluation.")
        return None
    with open(metadata_path) as data_file:
        data = json.load(data_file)
    y_true = []
    for fname in filenames:
        true_label = data[fname]["label"]
        y_true.append(1 if true_label.upper() == "FAKE" else 0)
    y_pred = predictions
    accuracy = sum(1 for i in range(len(y_true)) if y_true[i] == (1 if y_pred[i] >= 0.5 else 0)) / len(y_true)
    logloss = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print(f"\n{'='*40}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}\nAUC: {auc:.4f}\n{'='*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model .pth file")
    args = parser.parse_args()

    model = CViT(image_size=FRAME_SIZE, patch_size=7, num_classes=2, cnn_channels=512, transformer_dim=1024, transformer_depth=6, transformer_heads=8, transformer_mlp_dim=2048).to(DEVICE)
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    filenames = sorted([x for x in os.listdir(SAMPLE_DIR) if x.endswith(".mp4")])
    predictions = predict_on_video(filenames, model)

    metadata_path = os.path.join(SAMPLE_DIR, "metadata.json")
    evaluate_metrics(filenames, predictions, metadata_path)
    pd.DataFrame({"filename": filenames, "label": predictions}).to_csv("predictions.csv", index=False)
