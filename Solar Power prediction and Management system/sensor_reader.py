"""
sensor_reader.py — Irradiance Sensor Reader (RS485 Modbus RTU via COM Port)

Reads GHI from the irradiance sensor every 5 minutes.
Writes into solar_prediction.db — same DB the FastAPI app uses.

Run independently alongside the FastAPI server:
    python sensor_reader.py

Requires: pyserial  →  pip install pyserial
"""

import serial
import time
import struct
import sqlite3
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
PORT            = "COM11"       # Windows: COM10 | Linux: /dev/ttyUSB0
BAUD            = 4800
SAMPLE_INTERVAL = 300           # 5 minutes
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),

    "solar_predictions.db"
)
# Modbus RTU frame: Unit 01, FC 03, Reg 0x0000, Count 0x0001
REQUEST = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x84, 0x0A])


# ── CRC-16 / Modbus ───────────────────────────────────────────────────────────
def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def validate_crc(frame: bytes) -> bool:
    if len(frame) < 3:
        return False
    return struct.unpack("<H", frame[-2:])[0] == crc16_modbus(frame[:-2])


# ── DB Init ───────────────────────────────────────────────────────────────────
def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sensor_raw_readings (
            timestamp    TEXT PRIMARY KEY,
            ghi_wm2      REAL,
            crc_valid    INTEGER NOT NULL,
            raw_response TEXT
        );

        CREATE TABLE IF NOT EXISTS sensor_data (
            hour_timestamp TEXT PRIMARY KEY,
            ghi_avg        REAL,
            ghi_min        REAL,
            ghi_max        REAL,
            sample_count   INTEGER,
            completeness   REAL
        );
    """)
    conn.commit()
    return conn


# ── Writers ───────────────────────────────────────────────────────────────────
def insert_raw(conn, ts: datetime, ghi, crc_ok: bool, raw: bytes):
    conn.execute(
        "INSERT OR REPLACE INTO sensor_raw_readings VALUES (?, ?, ?, ?)",
        (ts.isoformat(), ghi, int(crc_ok), raw.hex() if raw else None)
    )
    conn.commit()


def insert_hourly(conn, hour_ts: datetime, samples: list):
    if not samples:
        return
    conn.execute(
        "INSERT OR REPLACE INTO sensor_data VALUES (?, ?, ?, ?, ?, ?)",
        (
            hour_ts.isoformat(),
            round(sum(samples) / len(samples), 3),
            round(min(samples), 3),
            round(max(samples), 3),
            len(samples),
            round(len(samples) / 12, 3),  # 12 samples/hour at 5-min cadence
        )
    )
    conn.commit()


def hour_floor(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    conn = init_db(DB_PATH)
    ser  = serial.Serial(port=PORT, baudrate=BAUD,
                         bytesize=8, parity='N', stopbits=1, timeout=1)

    print(f"[Sensor Reader]  DB: {DB_PATH}  Port: {PORT} @ {BAUD} baud")
    print(f"Polling every {SAMPLE_INTERVAL}s — Ctrl+C to stop\n")
    print("SENSOR WRITING DB:", DB_PATH)
    hourly_samples = []
    current_hour   = hour_floor(datetime.now())

    try:
        while True:
            loop_start = time.time()

            ser.write(REQUEST)
            time.sleep(1)
            response  = ser.read(7)
            timestamp = datetime.now().astimezone()

            # Hour rollover
            new_hour = hour_floor(timestamp)
            if new_hour != current_hour:
                # User convention: 3pm-4pm data is stored as 4pm
                insert_hourly(conn, new_hour, hourly_samples)
                n = len(hourly_samples)
                avg = sum(hourly_samples) / n if hourly_samples else 0
                print(f"\n── Hour {current_hour:%H:00}-{new_hour:%H:00} complete | stored as {new_hour:%H:00} | samples={n}/12 | avg={avg:.1f} W/m² ──\n")
                hourly_samples = []
                current_hour   = new_hour

            # Parse
            ghi    = None
            crc_ok = False
            if len(response) == 7:
                crc_ok = validate_crc(response)
                if crc_ok:
                    ghi = float(struct.unpack(">H", response[3:5])[0])
                    hourly_samples.append(ghi)

            status = f"GHI: {ghi:>6.1f} W/m²  CRC:✓" if ghi is not None else \
                     f"GHI: ------        CRC:✗  [{response.hex()}]" if len(response) == 7 else \
                     f"⚠ Short response ({len(response)} bytes)"
            print(f"{timestamp:%Y-%m-%d %H:%M:%S}  {status}")

            insert_raw(conn, timestamp, ghi, crc_ok, response)

            time.sleep(max(0, SAMPLE_INTERVAL - (time.time() - loop_start)))

    except KeyboardInterrupt:
        print("\nFlushing last partial hour...")
        # Use end-of-hour timestamp for the partial hour as well
        from datetime import timedelta
        insert_hourly(conn, current_hour + timedelta(hours=1), hourly_samples)

    finally:
        ser.close()
        conn.close()
        print("Closed.")


if __name__ == "__main__":
    main()
