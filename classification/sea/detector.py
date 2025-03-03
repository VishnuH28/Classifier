import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet
import time

SEA_CLASSES = {20: "sea"}  # ADE20K class index, adjust based on actual weights

def initialize_detector(model_path="efficientdet_d0_ade20k.h5"):
    try:
        print("Initializing EfficientDet-D0 detector for sea (ADE20K)...")
        detector = tf.keras.models.load_model(model_path, compile=False)
        print("EfficientDet-D0 detector initialized successfully!")
        return detector
    except Exception as e:
        print(f"Error initializing EfficientDet detector: {e}")
        return None

def preprocess_image(image_path, input_size=(512, 512)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = efficientnet.preprocess_input(img)
    return np.expand_dims(img, axis=0)

def detect_sea_single(detector, image_path, confidence_threshold=0.5):
    if detector is None:
        print("Detector not initialized!")
        return []

    try:
        img = preprocess_image(image_path)
        predictions = detector.predict(img)[0]
        boxes, scores, classes = predictions[:4], predictions[4], predictions[5]

        detected = []
        for i in range(int(predictions[6])):
            confidence = scores[i]
            class_id = int(classes[i])
            if class_id in SEA_CLASSES and confidence > confidence_threshold:
                y_min, x_min, y_max, x_max = boxes[i]
                h, w = img.shape[1:3]
                x, y = int(x_min * w), int(y_min * h)
                width, height = int((x_max - x_min) * w), int((y_max - y_min) * h)
                label = SEA_CLASSES[class_id]
                detected.append((label, (x, y, width, height), confidence))
        return detected
    except Exception as e:
        print(f"Error detecting sea in {image_path}: {e}")
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

    print("\nStarting sea detection with ADE20K...")
    start_time = time.time()
    stats = {'detected': 0, 'processed': 0, 'errors': 0, 'detections': {}}
    images_with_objects = []

    for image_path in image_files:
        try:
            detections = detect_sea_single(detector, image_path, confidence_threshold)
            stats['processed'] += 1
            if detections:
                images_with_objects.append(image_path)
                stats['detected'] += len(detections)
                stats['detections'][image_path] = detections
                print(f"Found {len(detections)} sea areas in {os.path.basename(image_path)}")

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

    print("\nSea Detection Complete!")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"Images per second: {images_per_second:.2f}")
    print(f"Total images processed: {stats['processed']}")
    print(f"Total sea areas detected: {stats['detected']}")
    print(f"Images with sea: {len(images_with_objects)} ({percentage_detected:.1f}%)")