import cv2
import numpy as np
import insightface
import logging
from typing import Tuple, List, Set
from pathlib import Path

# --- Configuration ---
DATASET_FOLDER = 'person_images'  # Folder with face images
OUTPUT_NPZ = 'face_embeddings.npz'  # Output file name
DETECTION_SIZE = (1024,1024)  # Higher resolution for better embeddings
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceDatasetCreator:
    """Creates face embeddings dataset from a folder of images."""
    
    def __init__(self, image_folder: str = DATASET_FOLDER):
        """Initialize with image folder path."""
        self.image_folder = Path(image_folder)
        self.face_model = None
        self.embeddings = []
        self.names = []
        
        self._initialize_face_model()
        self._validate_folder()

    def _initialize_face_model(self):
        """Initialize InsightFace model."""
        try:
            self.face_model = insightface.app.FaceAnalysis(name='buffalo_l')
            self.face_model.prepare(ctx_id=0, det_size=DETECTION_SIZE)
            logger.info("Face detection model initialized")
        except Exception as e:
            logger.error(f"Face model initialization failed: {e}")
            raise

    def _validate_folder(self):
        """Check if folder exists and contains valid images."""
        if not self.image_folder.is_dir():
            raise ValueError(f"Directory not found: {self.image_folder}")
            
        valid_files = [
            f for f in self.image_folder.iterdir() 
            if f.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        
        if not valid_files:
            raise ValueError(f"No valid images found in {self.image_folder}")
        logger.info(f"Found {len(valid_files)} valid images")

    def create_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process images and generate embeddings dataset.
        
        Returns:
            Tuple of (names, embeddings) arrays
        """
        processed_count = 0
        
        for img_path in self.image_folder.iterdir():
            if img_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
                
            try:
                if self._process_image(img_path):
                    processed_count += 1
            except Exception as e:
                logger.warning(f"Skipped {img_path.name}: {str(e)}")
        
        if processed_count == 0:
            raise RuntimeError("No valid faces processed")
            
        logger.info(f"Processed {processed_count} faces")
        return np.array(self.names), np.array(self.embeddings)

    def _process_image(self, img_path: Path) -> bool:
        """Process a single image and extract face embeddings."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("Failed to read image")
        
        faces = self.face_model.get(img)
        if len(faces) != 1:
            raise ValueError(f"Found {len(faces)} faces (expected 1)")
        
        self.embeddings.append(faces[0].embedding)
        self.names.append(img_path.stem)
        logger.debug(f"Processed {img_path.name}")
        return True

    def save_dataset(self, output_path: str = OUTPUT_NPZ):
        """Save embeddings and names to NPZ file."""
        names, embeddings = self.create_dataset()
        
        np.savez(
            output_path,
            names=names,
            embeddings=embeddings,
            all_students=np.unique(names)  # For compatibility with reference code
        )
        logger.info(f"Saved dataset to {output_path}")

def main():
    """Command-line interface matching original script's behavior."""
    try:
        creator = FaceDatasetCreator()
        creator.save_dataset()
        print(f"\nSuccessfully created dataset with {len(creator.names)} faces")
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error("Dataset creation failed", exc_info=True)

if __name__ == "__main__":
    main()