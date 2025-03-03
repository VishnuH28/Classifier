import cv2
import os
from ultralytics import YOLO
import time

ANIMAL_CLASSES = {
    15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep",
    20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe"
}

def initialize_detector(model_path="yolov8n.pt"):
    try:
        print("Initializing YOLOv8 Nano detector for animals (COCO)...")
        detector = YOLO(model_path)
        print("YOLOv8 Nano detector initialized successfully!")
        return detector
    except Exception as e:
        print(f"Error initializing YOLOv8 detector: {e}")
        return None

def detect_animals_single(detector, image_path, confidence_threshold=0.5):
    if detector is None:
        print("Detector not initialized!")
        return []

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image {image_path}")
            return []

        results = detector(img)
        detected = []

        for result in results:
            for box in result.boxes:
                x, y, w, h = map(int, box.xywh[0])
                confidence = box.conf.item()
                class_id = int(box.cls.item())
                if class_id in ANIMAL_CLASSES and confidence > confidence_threshold:
                    label = ANIMAL_CLASSES[class_id]
                    detected.append((label, (x, y, w, h), confidence))
        return detected
    except Exception as e:
        print(f"Error detecting animals in {image_path}: {e}")
        return []

def process_images(input_folder, output_folder=None, confidence_threshold=0.5, max_images=50):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        return [], {'detected': 0, 'processed': 0, 'errors': 0, 'detections': {}}

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    image_files = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in {'.jpg', '.jpeg', '.png'}
    ][:max_images]

    if not image_files:
        print("No valid images found.")
        return [], {'detected': 0, 'processed': 0, 'errors': 0, 'detections': {}}

    detector = initialize_detector()
    if not detector:
        return [], {'detected': 0, 'processed': 0, 'errors': 0, 'detections': {}}

    print("\nStarting animal detection with COCO...")
    start_time = time.time()
    stats = {'detected': 0, 'processed': 0, 'errors': 0, 'detections': {}}
    images_with_objects = []

    for image_path in image_files:
        try:
            detections = detect_animals_single(detector, image_path, confidence_threshold)
            stats['processed'] += 1
            if detections:
                images_with_objects.append(image_path)
                stats['detected'] += len(detections)
                stats['detections'][image_path] = detections
                print(f"Found {len(detections)} animals in {os.path.basename(image_path)}")

                if output_folder:
                    img = cv2.imread(image_path)
                    for label, (x, y, w, h), _ in detections:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    output_path = os.path.join(output_folder, os.path.basename(image_path))
                    cv2.imwrite(output_path, img)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            stats['errors'] += 1

    print_summary(start_time, stats, images_with_objects)
    return images_with_objects, stats

def print_summary(start_time, stats, images_with_objects):
    elapsed_time = time.time() - start_time
    avg_time_per_image = elapsed_time / stats['processed'] if stats['processed'] > 0 else 0
    images_per_second = stats['processed'] / elapsed_time if elapsed_time > 0 else 0
    percentage_detected = (len(images_with_objects) / stats['processed'] * 100) if stats['processed'] > 0 else 0

    print("\nAnimal Detection Complete!")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"Images per second: {images_per_second:.2f}")
    print(f"Total images processed: {stats['processed']}")
    print(f"Total animals detected: {stats['detected']}")
    print(f"Images with animals: {len(images_with_objects)} ({percentage_detected:.1f}%)")