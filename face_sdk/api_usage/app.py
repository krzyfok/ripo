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
                            feature = self.extract_features(str(img_file))
                            face_db[person_name].append(feature)
                        except Exception as e:
                            logger.warning(f"Błąd przetwarzania {img_file}: {str(e)}")
        return face_db

    def extract_features(self, image_path):
        """Ekstrakcja cech bez wizualizacji"""
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
        return self.face_recognizer.inference_on_image(cropped_image).flatten()

    def process_image(self, image_path):
        """Przetwarzanie obrazu z wizualizacją"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")

        # Detekcja twarzy
        dets = self.face_detector.inference_on_image(image)
        if dets.shape[0] == 0:
            return None, None, None

        # Rysowanie ramek
        image_with_boxes = image.copy()
        for box in dets:
            box = list(map(int, box))
            cv2.rectangle(image_with_boxes, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        # Ekstrakcja cech
        x, y, w, h = map(int, dets[0][:4])
        landmarks = self.face_aligner.inference_on_image(image, dets[0])
        landmarks_list = landmarks.astype(np.int32).flatten().tolist()
        cropped_image = self.face_cropper.crop_image_by_mat(image, landmarks_list)
        feature = self.face_recognizer.inference_on_image(cropped_image).flatten()

        return feature, image_with_boxes, (x, y, w, h)

    def recognize_face(self, image_path, threshold=0.5):
        """Główna funkcja rozpoznawania z wizualizacją"""
        try:
            feature, image_with_boxes, bbox = self.process_image(image_path)
            if feature is None:
                return "Nieznany", 0.0, image_with_boxes

            best_match = ("Nieznany", 0.0)
            for name, features in self.face_db.items():
                for ref_feature in features:
                    similarity = np.dot(feature, ref_feature)
                    if similarity > best_match[1]:
                        best_match = (name, similarity)

            # Dodanie tekstu z wynikiem
            x, y, w, h = bbox
            result_text = f"{best_match[0]}: {best_match[1]*100:.2f}%"
            cv2.putText(image_with_boxes, result_text, 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (0, 0, 255), 1)

            final_result = best_match if best_match[1] > threshold else ("Nieznany", 0.0)
            return final_result[0], final_result[1], image_with_boxes
            
        except Exception as e:
            logger.error(f"Błąd rozpoznawania: {str(e)}")
            return "Błąd", 0.0, None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='System rozpoznawania twarzy')
    parser.add_argument('image', help='Ścieżka do zdjęcia do analizy')
    args = parser.parse_args()

    recognizer = FaceRecognizer()
    name, confidence, result_image = recognizer.recognize_face(args.image)
    
    print(f"\nWynik rozpoznania: {name}")
    print(f"Pewność: {confidence*100:.2f}%\n")

    if result_image is not None:
        cv2.imshow('Wynik rozpoznania', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
