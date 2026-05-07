import csv
import os
import logging
from .db import get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_csv_to_db(csv_path: str):
    """
    Reads crop climate requirements from a CSV file and inserts/updates them in the database.
    
    Expected CSV Structure:
    Crop_Name, T_Max_C, T_Min_C, T_Optimum_C, Rainfall_mm, Altitude_m_MSL
    
    Args:
        csv_path (str): The absolute or relative path to the CSV file.
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}")
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    query = """
        INSERT INTO crop_thresholds (
            crop_name, temp_min, temp_max, temp_optimum, rainfall_expected, altitude_expected
        ) VALUES (
            %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (crop_name) DO UPDATE SET
            temp_min = EXCLUDED.temp_min,
            temp_max = EXCLUDED.temp_max,
            temp_optimum = EXCLUDED.temp_optimum,
            rainfall_expected = EXCLUDED.rainfall_expected,
            altitude_expected = EXCLUDED.altitude_expected;
    """

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            with open(csv_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                # Handling UTF-8 BOM if present
                if reader.fieldnames and reader.fieldnames[0].startswith('\ufeff'):
                    reader.fieldnames[0] = reader.fieldnames[0].replace('\ufeff', '')
                    
                count = 0
                for row in reader:
                    # Clean and parse data
                    crop_name = row.get('Crop_Name', '').strip()
                    if not crop_name:
                        continue
                        
                    t_max = float(row.get('T_Max_C', 0))
                    t_min = float(row.get('T_Min_C', 0))
                    t_optimum = float(row.get('T_Optimum_C', 0))
                    rainfall = float(row.get('Rainfall_mm', 0))
                    altitude = float(row.get('Altitude_m_MSL', 0))
                    
                    # Insert or update
                    cursor.execute(query, (crop_name, t_min, t_max, t_optimum, rainfall, altitude))
                    count += 1
            
            conn.commit()
            logger.info(f"Successfully imported/updated {count} crop records from {csv_path}.")
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Error importing CSV to database: {e}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    # Optional execution block to test loading the CSV independently
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_file_path = os.path.join(project_root, 'data', 'crop_climate_requirements.csv')
    import_csv_to_db(csv_file_path)
