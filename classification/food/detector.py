import cv2
import os
from ultralytics import YOLO
import time

# COCO food-related classes 
FOOD_CLASSES = {
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake"
}

def initialize_detector(model_path="yolov8n.pt"):
    """Initialize the YOLOv8 Nano detector."""
    try:
        print("Initializing YOLOv8 Nano detector for food...")
        detector = YOLO(model_path)
        print("YOLOv8 Nano detector initialized successfully!")
        return detector
    except Exception as e:
        print(f"Error initializing YOLOv8 detector: {e}")
        return None

def detect_food(detector, image_path, confidence_threshold=0.5):
    """Detect food items in a single image."""
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
                if class_id in FOOD_CLASSES and confidence > confidence_threshold:
                    label = FOOD_CLASSES[class_id]
                    detected.append((label, (x, y, w, h), confidence))
        return detected
    except Exception as e:
        print(f"Error detecting food in {image_path}: {e}")
        return []

def process_images(input_folder, output_folder=None, confidence_threshold=0.5):
    """Process images in a folder for food detection."""
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        return [], {'detected': 0, 'processed': 0, 'errors': 0}

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    image_files = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in {'.jpg', '.jpeg', '.png'}
    ]

    if not image_files:
        print("No valid images found.")
        return [], {'detected': 0, 'processed': 0, 'errors': 0}

    detector = initialize_detector()
    if not detector:
        return [], {'detected': 0, 'processed': 0, 'errors': 0}

    print("\nStarting food detection...")
    start_time = time.time()
    stats = {'detected': 0, 'processed': 0, 'errors': 0}
    images_with_objects = []

    for image_path in image_files:
        try:
            detections = detect_food(detector, image_path, confidence_threshold)
            stats['processed'] += 1
            if detections:
                images_with_objects.append(image_path)
                stats['detected'] += len(detections)
                print(f"Found {len(detections)} food items in {os.path.basename(image_path)}")

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
    """Print summary statistics for food detection."""
    elapsed_time = time.time() - start_time
    avg_time_per_image = elapsed_time / stats['processed'] if stats['processed'] > 0 else 0
    images_per_second = stats['processed'] / elapsed_time if elapsed_time > 0 else 0
    percentage_detected = (len(images_with_objects) / stats['processed'] * 100) if stats['processed'] > 0 else 0

    print("\nFood Detection Complete!")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"Images per second: {images_per_second:.2f}")
    print(f"Total images processed: {stats['processed']}")
    print(f"Total food items detected: {stats['detected']}")
    print(f"Images with food: {len(images_with_objects)} ({percentage_detected:.1f}%)")