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

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/TelloAIv0.1.git
    cd TelloAIv0.1
    ```

2. **Install required Python packages:**

    ```bash
    pip install opencv-contrib-python numpy
    ```

## Usage

1. **Prepare your video file:**
   
   Ensure you have the video file (`TelloAIv0.0_video.mp4`) in the project directory.

2. **Run the script:**

    ```bash
    python TelloAI.py
    ```

3. **Output:**

    - `aruco_detection_results.csv`: A CSV file containing the frame ID, marker ID, 2D corner points, and 3D pose (distance, yaw, pitch, roll) for each detected marker.
    - `annotated_aruco_video.mp4`: A video file with annotated frames showing the detected markers and their IDs.

## Code Explanation

### detect_aruco_codes(frame, aruco_dict, aruco_params)

This function detects ArUco markers in a given video frame and estimates their 3D position and orientation.


### estimate_aruco_3d_position(corners, camera_matrix, dist_coeffs)

This function uses the detected marker corners and camera calibration parameters to estimate the 3D position and orientation of the markers.

![detects ArUco markers](https://github.com/nirmeir/TelloAI-V.0.1/assets/24902621/1d89e151-d8e2-4461-9925-b8c0c71dc57b)

### Main Script

The script reads frames from the input video, processes each frame to detect ArUco markers, estimates their 3D pose, writes the results to a CSV file, and saves an annotated video.

## Assumptions

- The camera parameters are approximated for a 720p resolution video.
- The ArUco marker side length is assumed to be 0.05 meters (5 cm).



## Contact

For any questions or issues, please contact [your email].
