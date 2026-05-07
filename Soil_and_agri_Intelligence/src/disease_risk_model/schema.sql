CREATE TABLE IF NOT EXISTS crop_thresholds (
    id SERIAL PRIMARY KEY,
    crop_name VARCHAR(100) UNIQUE NOT NULL,
    temp_min FLOAT NOT NULL,
    temp_max FLOAT NOT NULL,
    temp_optimum FLOAT NOT NULL,
    rainfall_expected FLOAT NOT NULL,
    altitude_expected FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Trigger for updating updated_at column
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_crop_thresholds_modtime ON crop_thresholds;

CREATE TRIGGER update_crop_thresholds_modtime
    BEFORE UPDATE ON crop_thresholds
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();
