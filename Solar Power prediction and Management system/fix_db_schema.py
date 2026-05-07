import sqlite3
import pandas as pd

DB_PATH = 'solar_predictions.db'
models = ['xgboost', 'lightgbm', 'randomforest', 'extratrees', 'svr']

def fix_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for model in models:
        table_name = f'predictions_{model}'
        print(f"Fixing {table_name}...")
        
        # 1. Read existing data
        try:
            df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
            # 2. Deduplicate
            df = df.drop_duplicates(subset=['timestamp'])
            
            # 3. Rename old table
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}_old")
            cursor.execute(f"ALTER TABLE {table_name} RENAME TO {table_name}_old")
            
            # 4. Create new table with PRIMARY KEY
            cursor.execute(f'''
                CREATE TABLE {table_name} (
                    timestamp TEXT PRIMARY KEY,
                    predicted_ghi REAL
                )
            ''')
            
            # 5. Insert clean data
            df.to_sql(table_name, conn, if_exists='append', index=False)
            
            # 6. Drop old table
            cursor.execute(f"DROP TABLE {table_name}_old")
            print(f"  Done. {len(df)} unique rows preserved.")
            
        except Exception as e:
            print(f"  Error fixing {table_name}: {e}")
            
    conn.commit()
    conn.close()

if __name__ == '__main__':
    fix_schema()
