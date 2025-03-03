echo "Starting classifier server at port 8000"

THREADS=1
PORT=8000
TIMEOUT=300

gunicorn -w $THREADS -b 0.0.0.0:$PORT --timeout $TIMEOUT --log-level debug main:app