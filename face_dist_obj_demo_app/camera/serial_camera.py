# import time
# import serial
# import numpy as np
# import cv2

# # Initialize serial port on COM
# port = serial.Serial(
#     "COM6",
#     baudrate=115200,
#     parity=serial.PARITY_NONE,
#     stopbits=serial.STOPBITS_ONE,
#     bytesize=serial.EIGHTBITS,
#     timeout=1
# )
# port.reset_input_buffer()

# i = 1

# while True:
    
#     num_byte = 0
#     jpg_data = bytearray()
#     is_jpg = False

#     time.sleep(1)  

#     prev_byte = None
    
#     # Read from serial and save image
#     while True:
#         byte = port.read(1)

#         if not byte:  
#             break

#         if not is_jpg and prev_byte == b'\xff' and byte == b'\xd8':  # Start of JPEG
#             print("Found JPEG Header")
#             is_jpg = True
#             jpg_data.extend(prev_byte)  
#         if is_jpg:
#             jpg_data.extend(byte)
#             num_byte += 1

#             if prev_byte == b'\xff' and byte == b'\xd9':  # End of JPEG
#                 print("End of JPEG file")
#                 break

#         prev_byte = byte

#     if jpg_data:        
#         np_jpg = np.asarray(jpg_data, dtype=np.uint8)
#         image = cv2.imdecode(np_jpg, cv2.IMREAD_COLOR)
#         #i = 1
#         if image is not None:
            
#             image_path = f"face_dist_obj_demo_app\data\input\image{i}.jpg"
#             cv2.imwrite(image_path, image)
#             print(f"Image {i} saved at {image_path}")
            
#         else:
#             print("Failed to decode JPEG data.")
        
#         i += 1
    

# port.close()


import time
import serial
import numpy as np
import cv2
from pathlib import Path

# ------------------- Configuration -------------------
INPUT_DIR = Path("face_dist_obj_demo_app/input")
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize serial port
port = serial.Serial(
    "COM9",
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)
port.reset_input_buffer()

# ------------------- Helper -------------------
def get_next_image_index():
    existing_files = list(INPUT_DIR.glob("image*.jpg"))
    if not existing_files:
        return 1
    # Extract max index from existing filenames
    indices = [int(f.stem.replace("image", "")) for f in existing_files if f.stem.replace("image","").isdigit()]
    return max(indices) + 1 if indices else 1

i = get_next_image_index()

# ------------------- Read Loop -------------------
while True:
    num_byte = 0
    jpg_data = bytearray()
    is_jpg = False
    prev_byte = None
    
    time.sleep(1)  # small delay

    # Read bytes from serial
    while True:
        byte = port.read(1)
        if not byte:  
            break

        if not is_jpg and prev_byte == b'\xff' and byte == b'\xd8':  # JPEG header
            is_jpg = True
            jpg_data.extend(prev_byte)  

        if is_jpg:
            jpg_data.extend(byte)
            num_byte += 1
            if prev_byte == b'\xff' and byte == b'\xd9':  # JPEG end
                break

        prev_byte = byte

    # Save image if valid
    if jpg_data:
        np_jpg = np.frombuffer(jpg_data, dtype=np.uint8)
        image = cv2.imdecode(np_jpg, cv2.IMREAD_COLOR)

        if image is not None:
            image_path = INPUT_DIR / f"image{i}.jpg"
            cv2.imwrite(str(image_path), image)
            print(f"Image {i} saved at {image_path}")
            i += 1
        else:
            print("Failed to decode JPEG data.")

port.close()
