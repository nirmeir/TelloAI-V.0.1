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

# Function to generate movement commands based on QR code positions
def generate_movement_commands(target_qrs, current_qrs):
    commands = []

    for target in target_qrs:
        target_center = np.mean(target['points'], axis=0)
        target_area = target['area']
        current = next((c for c in current_qrs if c['id'] == target['id']), None)

        if current:
            current_center = np.mean(current['points'], axis=0)
            current_area = current['area']
            delta_x = target_center[0] - current_center[0]
            delta_y = target_center[1] - current_center[1]
            area_ratio = target_area / current_area if current_area > 0 else 1

            # Determine movement direction
            if abs(delta_x) > 50:  # Threshold for horizontal movement
                if delta_x > 0:
                    commands.append("right")
                else:
                    commands.append("left")

            if abs(delta_y) > 50:  # Threshold for vertical movement
                if delta_y > 0:
                    commands.append("down")
                else:
                    commands.append("up")

            if area_ratio > 1.1:  # Threshold for forward movement
                commands.append("forward")
            elif area_ratio < 0.9:  # Threshold for backward movement
                commands.append("backward")

            # Calculate rotation (simplified method)
            angle_diff = np.arctan2(current_center[1] - target_center[1], current_center[0] - target_center[0])
            if abs(angle_diff) > 0.1:  # Threshold for rotation
                if angle_diff > 0:
                    commands.append("turn-left")
                else:
                    commands.append("turn-right")

    if not commands:
        commands.append("hold")

    return commands

# Test movement command generation
def test_generate_movement_commands():
    # Load the target frame
    target_frame_path = 'target_frame.png'
    target_frame = cv2.imread(target_frame_path)

    if target_frame is None:
        print(f"Error: Could not open or find the image {target_frame_path}.")
        return

    target_qrs = detect_qr_codes(target_frame)

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_qrs = detect_qr_codes(frame)
        commands = generate_movement_commands(target_qrs, current_qrs)

        # Display the commands on the frame
        for i, command in enumerate(commands):
            cv2.putText(frame, command, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Live Video Command Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the test
test_generate_movement_commands()
