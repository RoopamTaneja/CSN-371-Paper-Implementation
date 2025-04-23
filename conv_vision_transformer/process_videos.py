import os
import json
import face_recognition
import cv2
import torch
import numpy as np
from utils.blazeface import BlazeFace

DATASET_DIR = "./sample_video_dataset/"
OUTPUT_DIR = "./extracted_faces/"
FRAMES_PER_VIDEO = 10
FACE_IMAGE_SIZE = 224
JPEG_QUALITY = 90


def load_metadata(meta_dir):
    """Load metadata.json from a directory, if present."""
    metafile = os.path.join(meta_dir, "metadata.json")
    if os.path.isfile(metafile):
        with open(metafile) as data_file:
            return json.load(data_file)
    return {}


def extract_faces_from_dataset(dataset_dir):
    """Extract faces from all videos in all subdirectories of the dataset."""
    for subdir in sorted(os.listdir(dataset_dir)):
        subdir_path = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        metadata = load_metadata(subdir_path)
        for filename in os.listdir(subdir_path):
            if filename.endswith(".mp4"):
                video_path = os.path.join(subdir_path, filename)
                label = "unknown"
                if filename in metadata and "label" in metadata[filename]:
                    label = metadata[filename]["label"].lower()
                save_dir = os.path.join(OUTPUT_DIR, subdir, label)
                os.makedirs(save_dir, exist_ok=True)
                print(f"Processing: {filename}")
                extract_and_save_faces(video_path, filename, save_dir)


class VideoFrameReader:
    """Helper class for reading frames from a video file."""

    def __init__(self, verbose=True, insets=(0, 0)):
        self.verbose = verbose
        self.insets = insets

    def read_random_frames(self, path, num_frames, seed=None):
        """Reads random frames from a video."""
        assert num_frames > 0
        np.random.seed(seed)
        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None

        frame_idxs = sorted(np.random.choice(np.arange(0, frame_count), num_frames))
        frames, idxs_read = [], []
        for idx in frame_idxs:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = capture.read()
            if not ret or frame is None:
                if self.verbose:
                    print(f"Error retrieving frame {idx} from {path}")
                break
            frames.append(self._postprocess_frame(frame))
            idxs_read.append(idx)
        capture.release()
        if frames:
            return np.stack(frames), idxs_read
        if self.verbose:
            print(f"No frames read from {path}")
        return None

    def _postprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.insets[0] > 0:
            W = frame.shape[1]
            p = int(W * self.insets[0])
            frame = frame[:, p:-p, :]
        if self.insets[1] > 0:
            H = frame.shape[0]
            q = int(H * self.insets[1])
            frame = frame[q:-q, :, :]
        return frame


class FaceExtractor:
    """Class for extracting faces from video frames using BlazeFace."""

    def __init__(self, video_read_fn, facedet):
        self.video_read_fn = video_read_fn
        self.facedet = facedet

    def process_video(self, video_path):
        """Extract faces from a single video."""
        input_dir = os.path.dirname(video_path)
        filenames = [os.path.basename(video_path)]
        return self._process_videos(input_dir, filenames, [0])

    def _process_videos(self, input_dir, filenames, video_idxs):
        target_size = self.facedet.input_size
        videos_read, frames_read, frames, tiles, resize_info = [], [], [], [], []

        for video_idx in video_idxs:
            filename = filenames[video_idx]
            video_path = os.path.join(input_dir, filename)
            result = self.video_read_fn(video_path)
            if result is None:
                continue
            videos_read.append(video_idx)
            my_frames, my_idxs = result
            frames.append(my_frames)
            frames_read.append(my_idxs)
            my_tiles, my_resize_info = self._tile_frames(my_frames, target_size)
            tiles.append(my_tiles)
            resize_info.append(my_resize_info)

        if not tiles:
            return []

        batch = np.concatenate(tiles)
        all_detections = self.facedet.predict_on_batch(batch, apply_nms=False)

        result, offs = [], 0
        for v in range(len(tiles)):
            num_tiles = tiles[v].shape[0]
            detections = all_detections[offs : offs + num_tiles]
            offs += num_tiles
            detections = self._resize_detections(detections, target_size, resize_info[v])
            num_frames = frames[v].shape[0]
            frame_size = (frames[v].shape[2], frames[v].shape[1])
            detections = self._untile_detections(num_frames, frame_size, detections)
            detections = self.facedet.nms(detections)
            for i in range(len(detections)):
                faces = self._add_margin_to_detections(detections[i], frame_size, 0.2)
                faces = self._crop_faces(frames[v][i], faces)
                scores = list(detections[i][:, 16].cpu().numpy())
                frame_dict = {
                    "video_idx": videos_read[v],
                    "frame_idx": frames_read[v][i],
                    "frame_w": frame_size[0],
                    "frame_h": frame_size[1],
                    "faces": faces,
                    "scores": scores,
                }
                result.append(frame_dict)
        return result

    def _tile_frames(self, frames, target_size):
        num_frames, H, W, _ = frames.shape
        split_size = min(H, W)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = 1
        num_h = 3 if W > H else 1
        splits = np.zeros((num_frames * num_v * num_h, target_size[1], target_size[0], 3), dtype=np.uint8)
        i = 0
        for f in range(num_frames):
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    crop = frames[f, y : y + split_size, x : x + split_size, :]
                    splits[i] = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                    x += x_step
                    i += 1
                y += y_step
        resize_info = [split_size / target_size[0], split_size / target_size[1], 0, 0]
        return splits, resize_info

    def _resize_detections(self, detections, target_size, resize_info):
        projected = []
        target_w, target_h = target_size
        scale_w, scale_h, offset_x, offset_y = resize_info
        for detection in detections:
            det = detection.clone()
            for k in range(2):
                det[:, k * 2] = (det[:, k * 2] * target_h - offset_y) * scale_h
                det[:, k * 2 + 1] = (det[:, k * 2 + 1] * target_w - offset_x) * scale_w
            for k in range(2, 8):
                det[:, k * 2] = (det[:, k * 2] * target_w - offset_x) * scale_w
                det[:, k * 2 + 1] = (det[:, k * 2 + 1] * target_h - offset_y) * scale_h
            projected.append(det)
        return projected

    def _untile_detections(self, num_frames, frame_size, detections):
        combined_detections = []
        W, H = frame_size
        split_size = min(H, W)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = 1
        num_h = 3 if W > H else 1
        i = 0
        for f in range(num_frames):
            detections_for_frame = []
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    detection = detections[i].clone()
                    if detection.shape[0] > 0:
                        for k in range(2):
                            detection[:, k * 2] += y
                            detection[:, k * 2 + 1] += x
                        for k in range(2, 8):
                            detection[:, k * 2] += x
                            detection[:, k * 2 + 1] += y
                    detections_for_frame.append(detection)
                    x += x_step
                    i += 1
                y += y_step
            combined_detections.append(torch.cat(detections_for_frame))
        return combined_detections

    def _add_margin_to_detections(self, detections, frame_size, margin=0.2):
        offset = torch.round(margin * (detections[:, 2] - detections[:, 0]))
        detections = detections.clone()
        detections[:, 0] = torch.clamp(detections[:, 0] - offset * 2, min=0)
        detections[:, 1] = torch.clamp(detections[:, 1] - offset, min=0)
        detections[:, 2] = torch.clamp(detections[:, 2] + offset, max=frame_size[1])
        detections[:, 3] = torch.clamp(detections[:, 3] + offset, max=frame_size[0])
        return detections

    def _crop_faces(self, frame, detections):
        faces = []
        for i in range(len(detections)):
            ymin, xmin, ymax, xmax = detections[i, :4].cpu().numpy().astype(int)
            face = frame[ymin:ymax, xmin:xmax, :]
            faces.append(face)
        return faces

    def keep_only_best_face(self, crops):
        """Keep only the best face per frame (highest score)."""
        for frame_data in crops:
            if len(frame_data["faces"]) > 0:
                frame_data["faces"] = frame_data["faces"][:1]
                frame_data["scores"] = frame_data["scores"][:1]


def extract_and_save_faces(video_path, filename, save_dir):
    """Extract faces from a video and save them as images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    facedet = BlazeFace().to(device)
    facedet.load_weights("utils/blazeface.pth")
    facedet.load_anchors("utils/anchors.npy")
    facedet.train(False)

    video_reader = VideoFrameReader()
    video_read_fn = lambda x: video_reader.read_random_frames(x, num_frames=FRAMES_PER_VIDEO)
    face_extractor = FaceExtractor(video_read_fn, facedet)

    faces = face_extractor.process_video(video_path)
    face_extractor.keep_only_best_face(faces)
    n = 0
    for frame_data in faces:
        for face in frame_data["faces"]:
            face_locations = face_recognition.face_locations(face)
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = face[top:bottom, left:right]
                resized_face = cv2.resize(face_image, (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
                out_name = f"{os.path.splitext(filename)[0]}_{n}.jpg"
                cv2.imwrite(os.path.join(save_dir, out_name), resized_face, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                n += 1
    if n == 0:
        print(f"No faces found in {filename}")


if __name__ == "__main__":
    extract_faces_from_dataset(DATASET_DIR)
    print("Completed extracting faces from all videos.")
