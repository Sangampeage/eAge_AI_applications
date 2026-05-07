import time
import serial
import numpy as np
import cv2
import os

# Initialize serial port on COM
port = serial.Serial(
    "COM6",
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)
port.reset_input_buffer()

i = 1

while True:
    
    num_byte = 0
    jpg_data = bytearray()
    is_jpg = False

    time.sleep(1)  

    prev_byte = None
    
    # Read from serial and save image
    while True:
        byte = port.read(1)

        if not byte:  
            break

        if not is_jpg and prev_byte == b'\xff' and byte == b'\xd8':  # Start of JPEG
            print("Found JPEG Header")
            is_jpg = True
            jpg_data.extend(prev_byte)  
        if is_jpg:
            jpg_data.extend(byte)
            num_byte += 1

            if prev_byte == b'\xff' and byte == b'\xd9':  # End of JPEG
                print("End of JPEG file")
                break

        prev_byte = byte

    if jpg_data:        
        np_jpg = np.asarray(jpg_data, dtype=np.uint8)
        image = cv2.imdecode(np_jpg, cv2.IMREAD_COLOR)
        #i = 1
        if image is not None:
            # Ensure the directory exists
            os.makedirs("Images", exist_ok=True)
            
            image_path = f"Images\\image{i}.jpg"
            cv2.imwrite(image_path, image)
            print(f"Image {i} saved at {image_path}")
            
        else:
            print("Failed to decode JPEG data.")
        
        i += 1
    

port.close()
