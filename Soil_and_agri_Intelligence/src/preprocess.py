import pandas as pd
import os

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
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    # Standardize names (lowercase, strip whitespace) to avoid duplicates
    unique_crops_cleaned = sorted(list(set(str(c).strip().lower() for c in unique_crops)))
    unique_soils_cleaned = sorted(list(set(str(s).strip().lower() for s in unique_soils)))

    print(f"--- Analysis Results ---")
    print(f"Total Unique Crops: {len(unique_crops_cleaned)}")
    print(f"List of Unique Crops:\n{unique_crops_cleaned}\n")

    print(f"Total Unique Soil Types: {len(unique_soils_cleaned)}")
    print(f"List of Unique Soil Types:\n{unique_soils_cleaned}\n")

    return unique_crops_cleaned, unique_soils_cleaned

if __name__ == "__main__":
    # Path to the data directory based on the project structure
    # This assumes the script is in src/ and data is in data/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    analyze_datasets(data_dir)
