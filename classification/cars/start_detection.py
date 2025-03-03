import os
import uuid
import asyncio
from .detector import process_images as detect_cars_in_folder
from data.err_msgs import ErrorMessages
from database.postgres import postgres, check_connection
from data.table_names import TableNames

async def start_detection(r_id, abs_path):
    """Main entry point to start car detection process."""
    success = False
    msg = ""
    req_id = "rqid-" + str(uuid.uuid4())

    global postgres
    postgres = check_connection(postgres)

    try:
        with postgres.cursor() as cur:
            cur.execute(
                f"INSERT INTO {TableNames.DETECTION_REQUEST.value} (req_id, r_id, category, status) VALUES (%s, %s, %s, %s)",
                (req_id, r_id, 'cars', 'processing')
            )
            postgres.commit()

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"The folder {abs_path} does not exist.")

        input_folder = abs_path
        output_folder = os.path.join(os.path.dirname(__file__), "detected_cars")
        images_with_cars, stats = detect_cars_in_folder(input_folder, output_folder)

        if not images_with_cars:
            raise Exception("No cars detected in the provided folder.")

        with postgres.cursor() as cur:
            for image_path, detections in stats['detections'].items():
                for label, _, confidence in detections:
                    cur.execute(
                        f"INSERT INTO {TableNames.DETECTED_OBJECTS.value} (req_id, image_path, object_label, confidence) VALUES (%s, %s, %s, %s)",
                        (req_id, image_path, label, confidence)
                    )
            cur.execute(
                f"UPDATE {TableNames.DETECTION_REQUEST.value} SET status = 'completed' WHERE req_id = %s",
                (req_id,)
            )
            postgres.commit()

        success = True
        msg = "Car detection process completed successfully"
    except Exception as e:
        print(f"start_car_detection(): {str(e)}")
        msg = str(e) or ErrorMessages.GENERIC_ERROR.value
        try:
            with postgres.cursor() as cur:
                cur.execute(
                    f"UPDATE {TableNames.DETECTION_REQUEST.value} SET status = 'stuck' WHERE req_id = %s",
                    (req_id,)
                )
                postgres.commit()
        except Exception as db_e:
            print(f"Failed to update status: {db_e}")
    finally:
        return {"success": success, "msg": msg, "req_id": req_id}

if __name__ == "__main__":
    result = asyncio.run(start_detection("test_r_id", "/mnt/storage/Classifier/dataset/val2017"))
    print(result)