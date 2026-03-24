from clearml import Dataset
from dotenv import load_dotenv

DATASET_NAME = "fatigue-raw-frames-v2"
DATASET_PROJECT = "FatigueSense"

load_dotenv()

# Create new dataset version, with the previous version as parent
ds_new = Dataset.create(
    dataset_name=DATASET_NAME,
    dataset_project=DATASET_PROJECT,
    dataset_tags=["raw_frames", "v2"],
)

# Images
ds_new.add_files(
    path="../dataset_1/images",
    dataset_path="vid_001",
)

# Start upload
ds_new.upload(show_progress=True)
ds_new.finalize()
