from flask import Blueprint, request, jsonify
import asyncio
from database.postgres import postgres, check_connection
from data.err_msgs import ErrorMessages
from classification.animals.start_detection import start_detection as detect_animals
from classification.food.start_detection import start_detection as detect_food
from classification.plants.start_detection import start_detection as detect_plants
from classification.mountains.start_detection import start_detection as detect_mountains
from data.table_names import TableNames

apiRoutes = Blueprint('apiRoutes', __name__)

CATEGORY_HANDLERS = {
    'animals': detect_animals,
    'food': detect_food,
    'plants': detect_plants,
    'mountains': detect_mountains
}

@apiRoutes.route('/process/start', methods=['POST'])
def start_process_route():
    success = False
    msg = ""
    req_id = None

    try:
        data = request.get_json()
        r_id = data.get('r_id')
        abs_path = data.get('abs_path')
        category = data.get('category')

        if not r_id or not abs_path or not category:
            msg = "r_id, abs_path, and category are required"
            raise ValueError(msg)
        
        if not isinstance(r_id, str) or not isinstance(abs_path, str) or not isinstance(category, str):
            msg = "r_id, abs_path, and category must be strings"
            raise ValueError(msg)

        if category not in CATEGORY_HANDLERS:
            msg = f"Invalid category: {category}. Supported: {', '.join(CATEGORY_HANDLERS.keys())}"
            raise ValueError(msg)

        handler = CATEGORY_HANDLERS[category]
        process_result = asyncio.run(handler(r_id, abs_path))
        req_id = process_result['req_id']
        success = process_result['success']
        msg = process_result['msg']
    except Exception as e:
        print(f"start_process_route(): {e}")
        msg = msg or ErrorMessages.GENERIC_ERROR.value
        return jsonify({"success": success, "msg": msg, "req_id": req_id}), 400
    return jsonify({"success": success, "msg": msg, "req_id": req_id}), 200

@apiRoutes.route('/process/status', methods=['POST'])
def status_process_route():
    success = False
    msg = ""
    status = "not found"

    try:
        data = request.get_json()
        req_id = data.get('req_id')

        if not req_id or not isinstance(req_id, str):
            msg = "req_id is required and must be a string"
            raise ValueError(msg)

        global postgres
        postgres = check_connection(postgres)
        with postgres.cursor() as cur:
            cur.execute(
                f"SELECT status FROM {TableNames.DETECTION_REQUEST.value} WHERE req_id = %s",
                (req_id,)
            )
            result = cur.fetchone()
            if result:
                status = result[0]
                success = True
                msg = f"Status for {req_id} retrieved"
            else:
                msg = f"Process {req_id} not found"
    except Exception as e:
        print(f"status_process_route(): {e}")
        msg = msg or ErrorMessages.GENERIC_ERROR.value
        return jsonify({"success": success, "msg": msg, "status": status}), 400
    return jsonify({"success": success, "msg": msg, "status": status}), 200