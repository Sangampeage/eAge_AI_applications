# import sqlite3
# from pathlib import Path

# DB_PATH = Path("vision.db")


# def get_connection():
#     return sqlite3.connect(DB_PATH, check_same_thread=False)


# def init_db():
#     conn = get_connection()
#     cursor = conn.cursor()

#     # Input images
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS images (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             filename TEXT UNIQUE,
#             path TEXT,
#             processed INTEGER DEFAULT 0,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     """)

#     # Face recognition results
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS face_results (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             input_image TEXT,
#             output_image TEXT,
#             result_json TEXT,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     """)

#     # Pose distance results
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS pose_results (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             input_image TEXT,
#             output_image TEXT,
#             distances TEXT,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     """)

#     # Object detection results
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS object_results (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             input_image TEXT,
#             output_image TEXT,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         )
#     """)

#     conn.commit()
#     conn.close()

# def image_exists(filename):
#     conn = get_connection()
#     cursor = conn.cursor()

#     cursor.execute(
#         "SELECT 1 FROM images WHERE filename = ?",
#         (filename,)
#     )

#     result = cursor.fetchone()
#     conn.close()
#     return result is not None


# def add_image(filename, path):
#     conn = get_connection()
#     cursor = conn.cursor()

#     cursor.execute(
#         "INSERT OR IGNORE INTO images (filename, path) VALUES (?, ?)",
#         (filename, path)
#     )

#     conn.commit()
#     conn.close()


# def is_processed(filename):
#     conn = get_connection()
#     cursor = conn.cursor()

#     cursor.execute(
#         "SELECT processed FROM images WHERE filename = ?",
#         (filename,)
#     )

#     row = cursor.fetchone()
#     conn.close()

#     return row is not None and row[0] == 1


# def mark_processed(filename):
#     conn = get_connection()
#     cursor = conn.cursor()

#     cursor.execute(
#         "UPDATE images SET processed = 1 WHERE filename = ?",
#         (filename,)
#     )

#     conn.commit()
#     conn.close()


# def add_face_result(input_image, output_image, result_json):
#     conn = get_connection()
#     cursor = conn.cursor()
#     cursor.execute(
#         "INSERT INTO face_results (input_image, output_image, result_json) VALUES (?, ?, ?)",
#         (input_image, output_image, result_json)
#     )
#     conn.commit()
#     conn.close()

# def add_pose_result(input_image, output_image, distances):
#     conn = get_connection()
#     cursor = conn.cursor()
#     cursor.execute(
#         "INSERT INTO pose_results (input_image, output_image, distances) VALUES (?, ?, ?)",
#         (input_image, output_image, str(distances))
#     )
#     conn.commit()
#     conn.close()

# def add_object_result(input_image, output_image):
#     conn = get_connection()
#     cursor = conn.cursor()
#     cursor.execute(
#         "INSERT INTO object_results (input_image, output_image) VALUES (?, ?)",
#         (input_image, output_image)
#     )
#     conn.commit()
#     conn.close()


# def clear_all_data():
#     conn = get_connection()
#     cursor = conn.cursor()

#     cursor.execute("DELETE FROM images, face_results, pose_results, object_results")
#     conn.commit()
#     conn.close()





import sqlite3
from pathlib import Path

DB_PATH = Path("vision2.db")


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # ------------------ INPUT IMAGES ------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS input_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            status TEXT DEFAULT 'NEW',  -- NEW | PROCESSING | DONE | FAILED
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ------------------ FACE RESULTS ------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_image_id INTEGER NOT NULL,
            output_image_path TEXT,
            faces_detected INTEGER,
            result_json TEXT,
            status TEXT DEFAULT 'SUCCESS',  -- SUCCESS | FAILED
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (input_image_id) REFERENCES input_images(id)
        )
    """)

    # ------------------ POSE RESULTS ------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pose_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_image_id INTEGER NOT NULL,
            output_image_path TEXT,
            persons_detected INTEGER,
            distances TEXT,
            status TEXT DEFAULT 'SUCCESS',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (input_image_id) REFERENCES input_images(id)
        )
    """)

    # ------------------ OBJECT RESULTS ------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS object_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_image_id INTEGER NOT NULL,
            output_image_path TEXT,
            objects_detected INTEGER,
            result_json TEXT,
            status TEXT DEFAULT 'SUCCESS',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (input_image_id) REFERENCES input_images(id)
        )
    """)

    conn.commit()
    conn.close()

def insert_input_image(filename, path):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO input_images (filename, path)
        VALUES (?, ?)
    """, (filename, path))

    conn.commit()
    conn.close()



def get_new_images():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, filename, path
        FROM input_images
        WHERE status = 'NEW'
        ORDER BY created_at ASC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def mark_processing(image_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE input_images
        SET status = 'PROCESSING',
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (image_id,))

    conn.commit()
    conn.close()



def mark_done(image_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE input_images
        SET status = 'DONE',
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (image_id,))

    conn.commit()
    conn.close()



def mark_failed(image_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE input_images
        SET status = 'FAILED',
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (image_id,))

    conn.commit()
    conn.close()




def clear_all_data():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM input_images")
    conn.commit()
    conn.close()


print("DB PATH:", DB_PATH.resolve())
