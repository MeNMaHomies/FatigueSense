from scripts.upload_dataset import upload_dataset

DATASET_NAME = "fatigue-region-labels"
DATASET_PROJECT = "FatigueSense"

# Change these values to match your dataset details
args = {
    "dataset_name": DATASET_NAME,
    "dataset_project": DATASET_PROJECT,
    "data_path": [],
    "dataset_tags": [],
    "dataset_upload": "/path/in/dataset/for/upload",
}

dataset_id = upload_dataset(**args)
print(f"Dataset ID: {dataset_id}")
