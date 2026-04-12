import os

import cv2
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ultralytics import YOLO

REPO_ID = "FatigueSense/fatigue-sense"
WEIGHTS_FILENAME = "best.pt"
DEFAULT_CONF_THRESHOLD = 0.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = {
    0: "Head",
    1: "Torso",
    2: "Mouth",
    3: "Eyes",
}


def load_model(repo_id=REPO_ID, filename=WEIGHTS_FILENAME) -> YOLO:
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(model_path)


def _append_labels_file(class_dir: str, class_name: str, sample_id: str) -> None:
    labels_path = os.path.join(class_dir, f"{class_name.lower()}_labels.txt")
    entry = f"{sample_id}\t\n"
    existing = set()
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            existing = {line.split("\t")[0] for line in f}
    if sample_id not in existing:
        with open(labels_path, "a") as f:
            f.write(entry)


def crop_detections(
    image_path: str,
    output_dir: str,
    model: YOLO,
    target_classes: list[int],
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> list[str]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    h, w = image.shape[:2]
    results = model(image, conf=conf_threshold, verbose=False)

    # Get base name without extension.
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    saved_paths = []
    for result in results:
        for _, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            if class_id not in target_classes:
                continue

            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

            # Get video name
            train_dir = os.path.dirname(image_path)
            video_dir = os.path.abspath(os.path.join(train_dir, "..", ".."))
            video_name = os.path.basename(video_dir)

            # Get x & y coordinates, ensuring they are within image bounds.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop the image and save it.
            cropped = image[y1:y2, x1:x2]
            sample_id = f"{video_name}_{base_name}"
            filename = f"{sample_id}.png"

            # Ensure the output directory for the class exists.
            # The structure will be something like below:
            # out_dir/
            # ├── class_name/
            # │   ├── images/
            # │   │   ├── {vid_name}_frame_01.png
            # │   │   ├── {vid_name}_frame_02.png
            # │   │   ├── {vid_name}_frame_03.png
            # │   │   └── {vid_name}_frame_n.png
            # │   ├── eyes.yaml              # Metadata & Class Index definitions
            # │   └── eyes_labels.txt        # The "Master List" for all samples
            output_path = os.path.join(output_dir, class_name, "images", filename)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            cv2.imwrite(output_path, cropped)
            _append_labels_file(
                os.path.join(output_dir, class_name), class_name, sample_id
            )
            saved_paths.append(output_path)

    return saved_paths


def process_dataset(
    data_dir: str,
    output_dir: str,
    model: YOLO,
    target_classes: list[int],
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> list[str]:
    """
    Processes a dataset of videos, cropping detected objects and saving them to an output directory.

    Args:
        data_dir: The root directory containing video folders.
        output_dir: The directory where cropped images will be saved, organized by class.
        model: The YOLO model used for object detection.
        target_classes: A list of class IDs to be cropped.
        conf_threshold: The confidence threshold for detections.
    """
    all_saved = []
    image_paths = []

    # Build a deterministic list first so tqdm can show an accurate total.
    for vid_folder in sorted(os.listdir(data_dir)):
        images_dir = os.path.join(data_dir, vid_folder, "images")
        if not os.path.isdir(images_dir):
            continue
        for root, _, files in os.walk(images_dir):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                image_paths.append(os.path.join(root, fname))

    for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
        saved = crop_detections(
            image_path, output_dir, model, target_classes, conf_threshold
        )
        all_saved.extend(saved)

    return all_saved


# SAMPLE USAGE
if __name__ == "__main__":
    model = load_model()
    paths = process_dataset(
        "./data", "datasets", model, target_classes=list(CLASS_NAMES.keys())
    )
    print(f"Saved {len(paths)} crops.")
