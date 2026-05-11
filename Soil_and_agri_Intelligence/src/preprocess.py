import pandas as pd
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_datasets(data_dir):
    # file1 = os.path.join(data_dir, "Crop recommendation dataset.csv")
    file2 = os.path.join(data_dir, "sensor_Crop_Dataset (1).csv")

    unique_crops = set()
    unique_soils = set()

    for file_path in [file2]:
        if os.path.exists(file_path):
            try:
                # Read only a few lines to check column names first if it's too large, but these are small
                df = pd.read_csv(file_path)
                
                # Check for crop columns (case insensitive matching for column names)
                crop_cols = [col for col in df.columns if col.strip().lower() in ['crop', 'crops', 'label']]
                for col in crop_cols:
                    unique_crops.update(df[col].dropna().unique())
                
                # Check for soil columns
                soil_cols = [col for col in df.columns if col.strip().lower() in ['soil', 'soil_type', 'soil type']]
                for col in soil_cols:
                    unique_soils.update(df[col].dropna().unique())
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                return {
                    "status": "failed",
                    "message": f"Error reading {file_path}: {e}",
                    "data": None
                }
        else:
            logger.error(f"File not found: {file_path}")
            return {
                "status": "failed",
                "message": f"File not found: {file_path}",
                "data": None
            }

    # Standardize names (lowercase, strip whitespace) to avoid duplicates
    unique_crops_cleaned = sorted(list(set(str(c).strip().lower() for c in unique_crops)))
    unique_soils_cleaned = sorted(list(set(str(s).strip().lower() for s in unique_soils)))

    result = {
        "status": "success",
        "message": "Datasets analyzed successfully.",
        "data": {
            "total_unique_crops": len(unique_crops_cleaned),
            "unique_crops": unique_crops_cleaned,
            "total_unique_soils": len(unique_soils_cleaned),
            "unique_soils": unique_soils_cleaned
        }
    }
    return result

if __name__ == "__main__":
    # Path to the data directory based on the project structure
    # This assumes the script is in src/ and data is in data/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    result = analyze_datasets(data_dir)
    logger.info("Analysis Results: %s", json.dumps(result, indent=2))
