import os
import kagglehub
import shutil
import json
from pathlib import Path

# --- CONFIGURATION ---
os.environ['KAGGLE_USERNAME'] = "YOUR_KAGGLE_USERNAME"
os.environ['KAGGLE_KEY'] = "YOUR_KAGGLE_API_KEY"

DATASET_HANDLE = "delayedkarma/impressionist-classifier-data"
LOCAL_EXPORT_DIR = Path("data") / "raw" / "van_gogh_dataset"
METADATA_FILE = "metadata.json"

# Define target classes
TARGET_ARTISTS = {
    "positives": ["Gogh"],
    "negatives": ["Gauguin", "Pisarro", "Monet", "Manet", "Degas"]
}

def get_dataset():
    """Checks for local cache before downloading."""
    print(f"🔍 Checking for dataset: {DATASET_HANDLE}")
    
    # kagglehub.dataset_download is smart: 
    # It checks ~/.cache/kagglehub first. If exists, it returns the path instantly.
    try:
        path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"✅ Dataset located at: {path}")
        return path
    except Exception as e:
        print(f"❌ Error accessing Kaggle: {e}")
        return None

def build_dataset(source_path):
    if not source_path:
        return

    # Ensure export directory exists
    LOCAL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Handle the nested structure of the Kaggle dataset
    search_base = Path(source_path) / "training" / "training"
    if not search_base.exists():
        search_base = Path(source_path)

    available_folders = [f for f in search_base.iterdir() if f.is_dir()]

    metadata = []

    for label_type, keywords in TARGET_ARTISTS.items():
        dest_folder = LOCAL_EXPORT_DIR / label_type
        dest_folder.mkdir(parents=True, exist_ok=True)

        for folder in available_folders:
            folder_name = folder.name
            # Check if folder name matches our keywords
            if any(key.lower() in folder_name.lower() for key in keywords):
                source_dir = folder
                files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.jpeg")) + list(source_dir.glob("*.png"))

                print(f"📦 Processing {len(files)} images from '{folder_name}'...")

                for i, img_path in enumerate(files):
                    new_filename = f"{folder_name.replace(' ', '_')}_{i}.jpg"
                    target_path = dest_folder / new_filename

                    # Only copy if it doesn't already exist in the local project
                    if not target_path.exists():
                        shutil.copy2(img_path, target_path)

                    metadata.append({
                        "file_path": f"{label_type}/{new_filename}",
                        "label": 1 if label_type == "positives" else 0,
                        "artist": folder_name
                    })

    # Save metadata
    with open(LOCAL_EXPORT_DIR / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    # Verification summary
    n_total = len(metadata)
    n_pos = sum(1 for m in metadata if m["label"] == 1)
    n_neg = n_total - n_pos

    print(f"\n🚀 Project ready! Dataset at: {LOCAL_EXPORT_DIR.resolve()}")
    print(f"📊 Total images in metadata: {n_total} (positives: {n_pos}, negatives: {n_neg})")


if __name__ == "__main__":
    src = get_dataset()
    build_dataset(src)