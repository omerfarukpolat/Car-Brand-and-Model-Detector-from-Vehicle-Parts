#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stanford Cars YOLOv8 Araç Marka-Model Tespit Sistemi

Bu sistem Stanford Cars dataset'i ile fine-tune edilmiş YOLOv8 modelini kullanarak
görüntülerdeki araçların marka ve modellerini tespit eder.

Kullanım:
    1. Model eğitimi: python stanford_cars_detector.py --train
    2. Görüntü tespiti: python stanford_cars_detector.py --predict image.jpg
    3. Video işleme: python stanford_cars_detector.py --video video.mp4
    4. Kamera demosu: python stanford_cars_detector.py --camera
    5. Batch işleme: python stanford_cars_detector.py --batch folder_path
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ultralytics warning'leri devre dışı bırak
os.environ["YOLO_VERBOSE"] = "False"

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics kütüphanesi bulunamadı. Yüklemek için: pip install ultralytics")
    sys.exit(1)


class StanfordCarsDetector:
    """Stanford Cars YOLOv8 Araç Marka-Model Tespit Sistemi"""

    def __init__(self, dataset_path="stanford_cars_yolo", model_size="n"):
        """
        Args:
            dataset_path: Stanford Cars dataset yolu
            model_size: YOLOv8 model boyutu (n, s, m, l, x)
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.model = None
        self.class_names = {}

        # Dosya yolları
        self.data_yaml = self.dataset_path / "data" / "dataset.yaml"
        self.classes_file = self.dataset_path / "data" / "classes.txt"

        # Model kayıt yolu
        self.model_save_path = Path("runs") / "train" / f"yolov8{model_size}_stanford_cars"

        self._load_class_names()

    def _load_class_names(self):
        """Sınıf isimlerini yükle"""
        try:
            # Önce YAML dosyasından yükle
            if self.data_yaml.exists():
                with open(self.data_yaml, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        self.class_names = data['names']
                        logger.info(f"YAML'dan {len(self.class_names)} sınıf yüklendi")
                        return

            # YAML yoksa classes.txt'den yükle
            if self.classes_file.exists():
                with open(self.classes_file, 'r', encoding='utf-8') as f:
                    class_list = [line.strip() for line in f.readlines() if line.strip()]
                    self.class_names = {i: name for i, name in enumerate(class_list)}
                    logger.info(f"classes.txt'den {len(self.class_names)} sınıf yüklendi")
                    return

            logger.warning("Sınıf dosyaları bulunamadı, varsayılan Stanford Cars sınıfları kullanılacak")
            self._load_default_classes()

        except Exception as e:
            logger.error(f"Sınıf isimleri yüklenirken hata: {e}")
            self._load_default_classes()

    def _load_default_classes(self):
        """Varsayılan Stanford Cars sınıf isimlerini yükle"""
        default_classes = [
            "AM General Hummer SUV 2000", "Acura RL Sedan 2012", "Acura TL Sedan 2012",
            "Acura TL Type-S 2008", "Acura TSX Sedan 2012", "Acura Integra Type R 2001",
            "Acura ZDX Hatchback 2012", "Aston Martin V8 Vantage Convertible 2012",
            "Aston Martin V8 Vantage Coupe 2012", "Aston Martin Virage Convertible 2012",
            "Aston Martin Virage Coupe 2012", "Audi RS 4 Convertible 2008",
            "Audi A5 Coupe 2012", "Audi TTS Coupe 2012", "Audi R8 Coupe 2012",
            "Audi V8 Sedan 1994", "Audi 100 Sedan 1994", "Audi 100 Wagon 1994",
            "Audi TT Hatchback 2011", "Audi S6 Sedan 2011", "Audi S5 Convertible 2012"
            # ... (diğer sınıflar)
        ]
        self.class_names = {i: name for i, name in enumerate(default_classes)}

    def train(self, epochs=100, batch_size=16, img_size=640, patience=20):
        """
        YOLOv8 modelini Stanford Cars dataset'i üzerinde eğit

        Args:
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            img_size: Görüntü boyutu
            patience: Early stopping sabır değeri
        """
        logger.info(f"YOLOv8{self.model_size} modeli eğitimi başlatılıyor...")

        # Dataset YAML dosyasını kontrol et
        if not self.data_yaml.exists():
            logger.error(f"Dataset YAML dosyası bulunamadı: {self.data_yaml}")
            return None

        try:
            # YOLOv8 modelini yükle
            model = YOLO(f"yolov8{self.model_size}.pt")

            # Eğitimi başlat
            results = model.train(
                data=str(self.data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                patience=patience,
                save=True,
                project="runs/train",
                name=f"yolov8{self.model_size}_stanford_cars",
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',
                lr0=0.01,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                verbose=True
            )

            # En iyi modeli yükle
            best_model_path = Path("runs/train") / f"yolov8{self.model_size}_stanford_cars" / "weights" / "best.pt"
            if best_model_path.exists():
                self.model = YOLO(str(best_model_path))
                logger.info(f"Model eğitimi tamamlandı. En iyi model: {best_model_path}")
                return str(best_model_path)
            else:
                logger.error("Eğitilmiş model dosyası bulunamadı")
                return None

        except Exception as e:
            logger.error(f"Model eğitimi sırasında hata: {e}")
            return None

    def load_model(self, model_path=None):
        """
        Eğitilmiş modeli yükle

        Args:
            model_path: Model dosya yolu (None ise otomatik bulur)
        """
        try:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info(f"Model yüklendi: {model_path}")
                return True

            # Otomatik model arama
            possible_paths = [
                Path("runs/train") / f"yolov8{self.model_size}_stanford_cars" / "weights" / "best.pt",
                Path("runs/train") / f"yolov8{self.model_size}_stanford_cars2" / "weights" / "best.pt",
                Path("best.pt"),
                Path(f"yolov8{self.model_size}_stanford_cars.pt")
            ]

            for path in possible_paths:
                if path.exists():
                    self.model = YOLO(str(path))
                    logger.info(f"Model otomatik olarak bulundu ve yüklendi: {path}")
                    return True

            # Hiçbiri bulunamazsa temel modeli yükle
            logger.warning("Eğitilmiş model bulunamadı, temel YOLOv8 modeli yükleniyor...")
            self.model = YOLO(f"yolov8{self.model_size}.pt")
            logger.warning("DİKKAT: Bu model Stanford Cars üzerinde eğitilmemiş!")
            return True

        except Exception as e:
            logger.error(f"Model yüklenirken hata: {e}")
            return False

    def _draw_detection(self, image, box, class_id, confidence):
        """Tespit kutusunu çiz"""
        x1, y1, x2, y2 = map(int, box)

        # Sınıf adını al
        class_name = self.class_names.get(class_id, f"Class_{class_id}")

        # Uzun isimleri kısalt
        if len(class_name) > 40:
            parts = class_name.split()
            if len(parts) >= 3:
                class_name = f"{parts[0]} {parts[1]} {parts[2]}"

        # Kutu ve etiket
        color = (0, 255, 0)  # Yeşil
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Etiket metni
        label = f"{class_name}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Metin boyutu
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Metin arka planı
        cv2.rectangle(image, (x1, y1 - text_height - 10),
                      (x1 + text_width + 5, y1), color, -1)

        # Metin
        cv2.putText(image, label, (x1 + 2, y1 - 5),
                    font, font_scale, (255, 255, 255), thickness)

    def predict_image(self, image_path, conf_threshold=0.25, save_result=False, show_result=True):
        """
        Tek bir görüntüde tespit yap

        Args:
            image_path: Görüntü dosya yolu
            conf_threshold: Güven eşik değeri
            save_result: Sonucu kaydet
            show_result: Sonucu göster
        """
        if self.model is None:
            logger.error("Model yüklenmemiş! Önce load_model() çağırın.")
            return None

        if not Path(image_path).exists():
            logger.error(f"Görüntü dosyası bulunamadı: {image_path}")
            return None

        try:
            # Tespit yap
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                verbose=False
            )

            # Görüntüyü yükle
            image = cv2.imread(image_path)
            original_image = image.copy()

            detections = []

            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()

                        # Tespit bilgisini kaydet
                        detection = {
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, f"Class_{class_id}"),
                            'confidence': confidence,
                            'bbox': xyxy.tolist()
                        }
                        detections.append(detection)

                        # Görüntü üzerine çiz
                        self._draw_detection(image, xyxy, class_id, confidence)

            # Sonuçları yazdır
            logger.info(f"Tespit edilen araç sayısı: {len(detections)}")
            for i, det in enumerate(detections):
                logger.info(f"  {i + 1}. {det['class_name']} (Güven: {det['confidence']:.2f})")

            # Görüntüyü göster
            if show_result:
                plt.figure(figsize=(15, 10))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"Stanford Cars Tespit - {Path(image_path).name}")
                plt.tight_layout()
                plt.show()

            # Sonucu kaydet
            if save_result:
                output_path = f"output_{Path(image_path).name}"
                cv2.imwrite(output_path, image)
                logger.info(f"Sonuç kaydedildi: {output_path}")

            return {
                'detections': detections,
                'annotated_image': image,
                'original_image': original_image
            }

        except Exception as e:
            logger.error(f"Tespit sırasında hata: {e}")
            return None

    def predict_video(self, video_path, conf_threshold=0.25, save_result=False):
        """
        Video dosyasında tespit yap

        Args:
            video_path: Video dosya yolu
            conf_threshold: Güven eşik değeri
            save_result: Sonucu kaydet
        """
        if self.model is None:
            logger.error("Model yüklenmemiş!")
            return None

        if not Path(video_path).exists():
            logger.error(f"Video dosyası bulunamadı: {video_path}")
            return None

        try:
            cap = cv2.VideoCapture(video_path)

            # Video özellikleri
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.release()

            logger.info(f"Video işleniyor: {width}x{height}, {fps:.1f} FPS, {total_frames} kare")

            # Video yazıcı
            output_path = None
            out = None

            if save_result:
                output_path = f"output_{Path(video_path).stem}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Stream modunda işle
            results = self.model.predict(
                source=video_path,
                conf=conf_threshold,
                stream=True,
                verbose=False
            )

            # İlerleme çubuğu
            pbar = tqdm(total=total_frames, desc="Video işleniyor")

            # FPS hesaplama
            frame_count = 0
            start_time = time.time()

            cv2.namedWindow("Stanford Cars - Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Stanford Cars - Video", 1280, 720)

            for r in results:
                frame_count += 1
                pbar.update(1)

                # Kareyi al
                frame = r.orig_img.copy()

                # Tespitleri çiz
                detections_count = 0
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()

                        self._draw_detection(frame, xyxy, class_id, confidence)
                        detections_count += 1

                # FPS bilgisi
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = time.time()
                else:
                    current_fps = fps

                # Bilgi metinleri
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Araçlar: {detections_count}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Kaydet
                if out is not None:
                    out.write(frame)

                # Göster
                cv2.imshow("Stanford Cars - Video", frame)

                # Çıkış kontrolü
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.close()

            if out is not None:
                out.release()
                logger.info(f"Video kaydedildi: {output_path}")

            cv2.destroyAllWindows()
            return output_path

        except Exception as e:
            logger.error(f"Video işleme hatası: {e}")
            return None

    def camera_demo(self, conf_threshold=0.25, camera_id=0):
        """
        Gerçek zamanlı kamera demosu

        Args:
            conf_threshold: Güven eşik değeri
            camera_id: Kamera ID'si
        """
        if self.model is None:
            logger.error("Model yüklenmemiş!")
            return

        try:
            cap = cv2.VideoCapture(camera_id)

            if not cap.isOpened():
                logger.error("Kamera açılamadı!")
                return

            logger.info("Kamera demosu başlatıldı. Çıkmak için 'q' tuşuna basın.")

            cv2.namedWindow("Stanford Cars - Kamera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Stanford Cars - Kamera", 1280, 720)

            fps_counter = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                fps_counter += 1

                # Tespit yap
                results = self.model.predict(frame, conf=conf_threshold, verbose=False)

                # Tespitleri çiz
                detections_count = 0
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        for box in r.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()

                            self._draw_detection(frame, xyxy, class_id, confidence)
                            detections_count += 1

                # FPS hesapla
                if fps_counter % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = time.time()
                else:
                    current_fps = 30  # Varsayılan

                # Bilgi metinleri
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Araçlar: {detections_count}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Stanford Cars - Kamera", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Kamera demosu hatası: {e}")

    def batch_predict(self, folder_path, conf_threshold=0.25, save_results=False):
        """
        Klasördeki tüm görüntülerde tespit yap

        Args:
            folder_path: Görüntü klasörü
            conf_threshold: Güven eşik değeri
            save_results: Sonuçları kaydet
        """
        if self.model is None:
            logger.error("Model yüklenmemiş!")
            return None

        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Klasör bulunamadı: {folder_path}")
            return None

        # Görüntü dosyalarını bul
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))

        if not image_files:
            logger.error("Klasörde görüntü dosyası bulunamadı!")
            return None

        logger.info(f"{len(image_files)} görüntü bulundu")

        results = []

        for image_file in tqdm(image_files, desc="Görüntüler işleniyor"):
            result = self.predict_image(
                str(image_file),
                conf_threshold=conf_threshold,
                save_result=save_results,
                show_result=False
            )

            if result:
                results.append({
                    'file': str(image_file),
                    'detections': result['detections']
                })

        # Özet rapor
        total_detections = sum(len(r['detections']) for r in results)
        logger.info(f"\nÖzet Rapor:")
        logger.info(f"İşlenen görüntü: {len(results)}")
        logger.info(f"Toplam tespit: {total_detections}")

        return results


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Stanford Cars YOLOv8 Araç Tespit Sistemi")

    # Model ayarları
    parser.add_argument("--dataset", default="stanford_cars_yolo",
                        help="Dataset yolu")
    parser.add_argument("--model-size", choices=['n', 's', 'm', 'l', 'x'],
                        default='n', help="Model boyutu")
    parser.add_argument("--model", help="Önceden eğitilmiş model yolu")

    # İşlem türü
    parser.add_argument("--train", action="store_true",
                        help="Model eğitimi")
    parser.add_argument("--predict", help="Görüntü tespiti")
    parser.add_argument("--video", help="Video işleme")
    parser.add_argument("--camera", action="store_true",
                        help="Kamera demosu")
    parser.add_argument("--batch", help="Klasör işleme")

    # Eğitim parametreleri
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch boyutu")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Görüntü boyutu")

    # Tespit parametreleri
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Güven eşik değeri")
    parser.add_argument("--save", action="store_true",
                        help="Sonuçları kaydet")
    parser.add_argument("--no-show", action="store_true",
                        help="Sonuçları gösterme")

    args = parser.parse_args()

    # Detector'ı başlat
    detector = StanfordCarsDetector(args.dataset, args.model_size)

    # Eğitim
    if args.train:
        detector.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
        return

    # Model yükle
    if not detector.load_model(args.model):
        logger.error("Model yüklenemedi!")
        return

    # İşlemleri gerçekleştir
    if args.predict:
        detector.predict_image(
            args.predict,
            conf_threshold=args.conf,
            save_result=args.save,
            show_result=not args.no_show
        )

    elif args.video:
        detector.predict_video(
            args.video,
            conf_threshold=args.conf,
            save_result=args.save
        )

    elif args.camera:
        detector.camera_demo(conf_threshold=args.conf)

    elif args.batch:
        detector.batch_predict(
            args.batch,
            conf_threshold=args.conf,
            save_results=args.save
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()