Created by Nir Meir, Ashraf Hijazi, Abed Massarwa

# TelloAIv0.1 - ArUco Marker Detection for Indoor Autonomous Drones

## Project Overview

This project is part of the Course of Autonomous Robotics at Ariel University. The goal is to detect ArUco markers in video frames captured by Tello Drones, estimate their 3D position and orientation, and annotate the video with this information.

The project includes:
1. Detecting ArUco markers in each frame of a video.
2. Estimating the 3D position and orientation (distance, yaw, pitch, roll) of each detected marker.
3. Writing the results to a CSV file.
4. Annotating the video frames with the detected markers and their IDs.
5. Ensuring real-time processing performance.

## Requirements

- Python 3.7+
- OpenCV (opencv-contrib-python)
- NumPy
- qrcode[pil]
- pyzbar

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/TelloAIv0.1.git
    cd TelloAIv0.1
    ```

2. **Install required Python packages:**

    ```bash
    pip install opencv-contrib-python numpy qrcode[pil] pyzbar
    ```

## Usage

### Part A: ArUco Marker Detection

1. **Prepare your video file:**
   
   Ensure you have the video file (`TelloAIv0.0_video.mp4`) in the project directory.

2. **Run the script:**

    ```bash
    python TelloAI.py
    ```

3. **Output:**

    - `aruco_detection_results.csv`: A CSV file containing the frame ID, marker ID, 2D corner points, and 3D pose (distance, yaw, pitch, roll) for each detected marker.
    - `annotated_aruco_video.mp4`: A video file with annotated frames showing the detected markers and their IDs.

Image example of the detection:
![detects ArUco markers](https://github.com/nirmeir/TelloAI-V.0.1/assets/24902621/1d89e151-d8e2-4461-9925-b8c0c71dc57b)

### Code Explanation

#### detect_aruco_codes(frame, aruco_dict, aruco_params)

This function detects ArUco markers in a given video frame and estimates their 3D position and orientation.

```python
def detect_aruco_codes(frame, aruco_dict, aruco_params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    aruco_data = []

    if ids is not None:
        for i in range(len(ids)):
            aruco_id = ids[i][0]
            pts = corners[i][0]

            # Estimate 3D position
            aruco_2d = pts
            aruco_3d = estimate_aruco_3d_position(pts, camera_matrix, dist_coeffs)

            aruco_data.append({
                'id': aruco_id,
                '2d_points': aruco_2d,
                '3d_info': aruco_3d
            })

    return aruco_data


```

### Part B: Live Video QR Code Detection and Movement Command Generation

### Usage

1. **Detect QR codes in live video:**

    ```python
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
    ```

2. **Generate movement commands based on detected QR codes:**

    ```python
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
        target_frame_path = 'target_frame_test.png'
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
 Image exammple of the detection and the movement commands:
![movement_commands test](https://github.com/nirmeir/TelloAI-V.0.1/assets/24902621/4e6d925d-e0aa-4b3a-9795-3ba3de7b9d60)

    # Run the test
    test_generate_movement_commands()
    ```
