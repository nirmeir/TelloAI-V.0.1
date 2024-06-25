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
### Part B: QR Code Detection and Movement Command Generation

1. **Generate and save the target frame with QR codes:**

    ```python
    import cv2
    import numpy as np
    import qrcode

    # Function to generate a QR code image
    def generate_qr_code(data, size=100):
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white').convert('RGB')
        img = img.resize((size, size), resample=cv2.INTER_AREA)
        return np.array(img)

    # Create a blank target frame image
    target_frame = np.ones((720, 960, 3), dtype=np.uint8) * 255  # White background

    # Generate and place QR codes on the target frame
    qr1 = generate_qr_code("1")
    qr2 = generate_qr_code("2")
    qr3 = generate_qr_code("3")

    # Define positions for the QR codes
    positions = [(50, 50), (300, 50), (175, 300)]  # Top-left corners

    # Place QR codes on the target frame
    for i, (x, y) in enumerate(positions):
        qr_img = [qr1, qr2, qr3][i]
        target_frame[y:y+qr_img.shape[0], x:x+qr_img.shape[1]] = qr_img

    # Save the target frame image
    cv2.imwrite('target_frame.png', target_frame)

    # Display the target frame image
    cv2.imshow('Target Frame', target_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

2. **Detect QR codes and generate movement commands:**

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

    # Load the target frame
    target_frame_path = 'target_frame.png'
    target_frame = cv2.imread(target_frame_path)

    if target_frame is None:
        print(f"Error: Could not open or find the image {target_frame_path}.")
        exit()

    target_qrs = detect_qr_codes(target_frame)

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

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
        cv2.imshow('Live Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    ```

## Assumptions

- The camera parameters are approximated for a 720p resolution video.
- The ArUco marker side length is assumed to be 0.05 meters (5 cm).

## Contact

For any questions or issues, please contact [your email].
