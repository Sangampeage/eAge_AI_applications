import serial
import os
import io
from datetime import datetime
from PIL import Image

# ================= CONFIG =================
PORT = "COM9"
BAUDRATE = 115200
READ_SIZE = 10240
IMAGE_DIR = "input"
MAX_IMAGE_SIZE = 250 * 1024
ENABLE_STREAM_SANITIZER = True
# ==========================================

JPEG_START = b'\xFF\xD8'
JPEG_END   = b'\xFF\xD9'

os.makedirs(IMAGE_DIR, exist_ok=True)

print(f"[INIT] Opening UART {PORT} @ {BAUDRATE}")
ser = serial.Serial(PORT, BAUDRATE, timeout=1)
ser.reset_input_buffer()
ser.reset_output_buffer()


# ---------- RECEIVER STATE ----------
rx_buffer = bytearray()
jpeg_data = bytearray()
receiving = False

img_count = 0
abort_count = 0
chunk_count = 0   # <<< NEW

print("[INFO] Waiting for incoming data...\n")

def trim_overlap(existing, incoming, max_check=128):
    """
    Removes duplicate/overlapping bytes between
    end of existing buffer and start of incoming data.
    """
    if not existing or not incoming:
        return incoming

    max_check = min(max_check, len(existing), len(incoming))

    for i in range(max_check, 0, -1):
        if existing[-i:] == incoming[:i]:
            print(f"[SANITIZER] Overlap detected: {i} bytes trimmed")
            return incoming[i:]

    return incoming

while True:
    raw = ser.read(READ_SIZE)
    if not raw:
        continue

    # ---------- CHUNK DEBUG ----------
    chunk_count += 1
    print(f"[CHUNK] #{chunk_count} received ({len(raw)} bytes)")

        # ---------- STREAM SANITIZER ----------
    if ENABLE_STREAM_SANITIZER:
        raw = trim_overlap(rx_buffer, raw)

    # Treat UART as pure byte stream
    rx_buffer.extend(raw)

    i = 0
    while i < len(rx_buffer):

        # ---------- WAIT FOR JPEG START ----------
        if not receiving:
            # ---- CASE-1: JPEG END before START ----
            if not receiving and rx_buffer[i:i+2] == JPEG_END:
                print(f"[ABORT] JPEG END found before START (chunk #{chunk_count})")
                abort_count += 1
                print(f"[STATS] Images saved: {img_count}, Aborted: {abort_count}\n")

                # Do NOT advance i
                # Do NOT change state
                # Just ignore and keep scanning
                i += 1
                continue


            # VALID JPEG START
            if rx_buffer[i:i+2] == JPEG_START:
                receiving = True
                jpeg_data.clear()
                jpeg_data.extend(JPEG_START)
                print(f"[DETECT] JPEG START at RX index {i} (chunk #{chunk_count})")
                i += 2
            else:
                i += 1

        # ---------- RECEIVING JPEG ----------
        else:
            # ---- CASE-2: JPEG START found before END ----
            if rx_buffer[i:i+2] == JPEG_START:
                print("[ABORT] JPEG END not found, new JPEG START detected")
                abort_count += 1
                print(f"[STATS] Images saved: {img_count}, Aborted: {abort_count}\n")
                jpeg_data.clear()
                receiving = False
                continue
                # jpeg_data.extend(JPEG_START)
                # receiving = True
                # i += 2
                # continue

            jpeg_data.append(rx_buffer[i])

            # ---- IMAGE SIZE LIMIT (SAFETY) ----
            if len(jpeg_data) > MAX_IMAGE_SIZE:
                print("[ABORT] Image exceeded size limit")
                abort_count += 1
                print(f"[STATS] Images saved: {img_count}, Aborted: {abort_count}\n")
                receiving = False
                jpeg_data.clear()
                break

            # ---- JPEG END FOUND ----
            if jpeg_data[-2:] == JPEG_END:
                # payload = jpeg_data[2:-2]
                core = jpeg_data[2:-2]   # remove FF D8 and FF D9
                while core and core[0] == 0xFF:
                    core = core[1:]

                payload = core

                # ---- CASE-3: ZERO-DATA IMAGE ----
                if len(payload) == 0:
                    print("[ABORT] JPEG has ZERO payload")
                    print(f"[DATA] Raw JPEG bytes: {jpeg_data.hex()}")
                    abort_count += 1
                    print(f"[STATS] Images saved: {img_count}, Aborted: {abort_count}\n")
                    jpeg_data.clear()
                    receiving = False
                    break

                # ---- CASE-4: PAYLOAD FULL OF ZEROS ----
                # print(f"[DEBUG] Payload bytes = {[hex(b) for b in payload]}")

                if payload and all(b == 0x00 for b in payload):
                    print("[ABORT] JPEG payload is FULL OF ZEROS")
                    print(f"[INFO] Zero bytes count: {len(payload)}")
                    print(f"[DATA] Payload sample (first 64B): {payload[:64].hex()} ...")
                    abort_count += 1
                    jpeg_data.clear()
                    receiving = False
                    break 

                receiving = False
                img_count += 1

                end_start_index = i - 1
                end_end_index   = i

                print(f"[DETECT] JPEG END at RX index {end_start_index}-{end_end_index} (chunk #{chunk_count})")
                print(f"[INFO] JPEG size: {len(jpeg_data)} bytes")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{timestamp}_{img_count:04d}.jpg"
                path = os.path.join(IMAGE_DIR, filename)

                with open(path, "wb") as f:
                    f.write(jpeg_data)

                print(f"[SAVE] Image saved → {path}")
                print(f"[STATS] Images saved: {img_count}, Aborted: {abort_count}\n")

                try:
                    Image.open(io.BytesIO(jpeg_data)).show()
                except Exception as e:
                    print("[WARN] Image display failed:", e)

                jpeg_data.clear()
                i += 1
                break

            i += 1

    # Clear RX buffer safely (current design assumption: slow chunks)
    rx_buffer.clear()
    # rx_buffer = rx_buffer[i:]


