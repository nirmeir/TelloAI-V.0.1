import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Function to detect QR codes and return their positions and sizes
def detect_qr_codes(frame):
    qr_codes = decode(frame)
    qr_data = []

    for qr in qr_codes:
        qr_id = qr.data.decode('utf-8')
        points = qr.polygon

        if len(points) == 4:
            pts = np.array([(point.x, point.y) for point in points], dtype=np.float32)
            area = cv2.contourArea(pts)
            qr_data.append({
                'id': qr_id,
                'points': pts,
                'area': area
            })

    return qr_data

# Test QR code detection in live video
def test_detect_qr_codes_live():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        qr_data = detect_qr_codes(frame)
        for qr in qr_data:
            pts = qr['points']
            pts = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, qr['id'], tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Live Video Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the test
test_detect_qr_codes_live()
