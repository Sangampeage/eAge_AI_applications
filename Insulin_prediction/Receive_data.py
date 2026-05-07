# import serial
# from datetime import datetime

# OUTPUT_FILE = "glucose_raw_log.txt"

# try:
#     with serial.Serial('COM9', 115200, timeout=2) as ser:
#         print(f"✅ Port opened successfully: {ser.port}")
        
#         while True:
#             # Read a line (assuming the device sends newline-terminated numbers)
#             line = ser.readline().decode(errors='ignore').strip()
            
#             if line:
#                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
#                 # Print for debugging
#                 print(f"{timestamp} | Received: {line}")
                
#                 # Append to file
#                 with open(OUTPUT_FILE, "a") as f:
#                     f.write(f"{timestamp} | {line}\n")

# except Exception as e:
#     print("❌ Failed to open port or read data:", e)




import serial
import json
from datetime import datetime

OUTPUT_FILE = "glucose_data.json"

try:
    with serial.Serial('COM9', 115200, timeout=2) as ser:
        print(f"✅ Port opened successfully: {ser.port}")
        
        while True:
            # Read a line (assuming the device sends newline-terminated numbers)
            line = ser.readline().decode(errors='ignore').strip()
            
            if line:
                try:
                    # Convert to float (works for both int and float values)
                    glucose_value = float(line)
                except ValueError:
                    print(f"⚠️ Skipped non-numeric data: {line}")
                    continue

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                record = {
                    "time": timestamp,
                    "Glucose": glucose_value
                }
                
                # Print for debugging
                print("Received:", record)
                
                # Append JSON object to file immediately
                with open(OUTPUT_FILE, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()  # ensure data is written to disk immediately

except Exception as e:
    print("❌ Failed to open port or read data:", e)
