"""
@author: [Twój Nick]
@modified: [Data]
@desc: Rozszerzenie oryginalnego skryptu o funkcjonalność rozpoznawania twarzy z bazy
"""
import sys
import os
import logging
import logging.config
import yaml
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.append('.')

# Konfiguracja logowania
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

# Importy z FaceX-Zoo
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

class FaceRecognizer:
    def __init__(self):
        self.initialize_models()
        self.face_db = self.load_face_database()

    def initialize_models(self):
        """Inicjalizacja wszystkich modeli"""
        model_path = 'models'
        scene = 'non-mask'

        # Inicjalizacja modelu detekcji twarzy
        logger.info('Ładowanie modelu detekcji twarzy...')
        face_det_loader = FaceDetModelLoader(model_path, 'face_detection', model_conf[scene]['face_detection'])
        det_model, det_cfg = face_det_loader.load_model()
        self.face_detector = FaceDetModelHandler(det_model, 'cpu', det_cfg)

        # Inicjalizacja modelu punktów charakterystycznych
        logger.info('Ładowanie modelu punktów charakterystycznych...')
        face_align_loader = FaceAlignModelLoader(model_path, 'face_alignment', model_conf[scene]['face_alignment'])
        align_model, align_cfg = face_align_loader.load_model()
        self.face_aligner = FaceAlignModelHandler(align_model, 'cpu', align_cfg)

        # Inicjalizacja modelu rozpoznawania
        logger.info('Ładowanie modelu rozpoznawania...')
        face_rec_loader = FaceRecModelLoader(model_path, 'face_recognition', model_conf[scene]['face_recognition'])
        rec_model, rec_cfg = face_rec_loader.load_model()
        self.face_recognizer = FaceRecModelHandler(rec_model.module.cpu(), 'cpu', rec_cfg)

        self.face_cropper = FaceRecImageCropper()

    # Reszta metod bez zmian


    def load_face_database(self):
        """Ładowanie bazy twarzy z folderu my_faces"""
        face_db = defaultdict(list)
        db_path = Path("api_usage/my_faces")
        
        if not db_path.exists():
            logger.error(f"Folder z bazą twarzy nie istnieje: {db_path}")
            sys.exit(-1)

        for person_name in os.listdir(db_path):
            person_dir = db_path / person_name
            if person_dir.is_dir():
                logger.info(f"Ładowanie zdjęć dla: {person_name}")
                for img_file in person_dir.glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            feature = self.process_image(str(img_file))
                            face_db[person_name].append(feature)
                        except Exception as e:
                            logger.warning(f"Błąd przetwarzania {img_file}: {str(e)}")
        return face_db

    def process_image(self, image_path):
        """Przetwarzanie pojedynczego obrazu"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")

        # Detekcja twarzy
        dets = self.face_detector.inference_on_image(image)
        if dets.shape[0] == 0:
            raise ValueError("Nie znaleziono twarzy na obrazie!")

        # Dopasowanie punktów i przycięcie
        landmarks = self.face_aligner.inference_on_image(image, dets[0])
        landmarks_list = landmarks.astype(np.int32).flatten().tolist()
        cropped_image = self.face_cropper.crop_image_by_mat(image, landmarks_list)

        # Ekstrakcja cech
        return self.face_recognizer.inference_on_image(cropped_image)

    def recognize_face(self, image_path, threshold=0.5):
        """Główna funkcja rozpoznawania"""
        try:
            query_feature = self.process_image(image_path)
            best_match = ("Nieznany", 0.0)

            for name, features in self.face_db.items():
                for ref_feature in features:
                    similarity = np.dot(query_feature, ref_feature)
                    if similarity > best_match[1]:
                        best_match = (name, similarity)

            return best_match if best_match[1] > threshold else ("Nieznany", 0.0)
            
        except Exception as e:
            logger.error(f"Błąd rozpoznawania: {str(e)}")
            return ("Błąd przetwarzania", 0.0)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='System rozpoznawania twarzy')
    parser.add_argument('image', help='Ścieżka do zdjęcia do analizy')
    args = parser.parse_args()

    recognizer = FaceRecognizer()
    name, confidence = recognizer.recognize_face(args.image)
    
    print(f"\nWynik rozpoznania: {name}")
    print(f"Pewność: {confidence:.4f}\n")
