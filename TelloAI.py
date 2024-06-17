import cv2
import numpy as np
import csv

# Camera parameters for 720p resolution
focal_length = 700  # A rough estimate for focal length
center = (960 // 2, 720 // 2)  # Assuming the principal point is at the center of the image

camera_matrix = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

# Function to detect ArUco codes in a frame
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

# Function for 3D estimation
def estimate_aruco_3d_position(corners, camera_matrix, dist_coeffs):
    # Define the ArUco marker side length (replace with your marker's actual size)
    marker_length = 0.05  # in meters

    # Estimate pose of each marker
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners], marker_length, camera_matrix, dist_coeffs)

    if rvecs is not None and tvecs is not None:
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Compute yaw, pitch, and roll
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        roll = np.arctan2(R[2, 1], R[2, 2])

        # Return distance and angles
        dist = np.linalg.norm(tvec)
        return dist, yaw, pitch, roll
    else:
        return None, None, None, None

# Load video
video_path = 'challengeB.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video FPS: {fps}, Frame Count: {frame_count}, Frame Width: {frame_width}, Frame Height: {frame_height}")

# Initialize ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Read and process frames
frame_id = 0
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    aruco_data = detect_aruco_codes(frame, aruco_dict, aruco_params)

    for aruco in aruco_data:
        result = [frame_id, aruco['id']]
        result.extend(aruco['2d_points'].flatten().tolist())
        result.extend(aruco['3d_info'])
        results.append(result)

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Write results to CSV
with open('aruco_detection_results.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Frame ID', 'ArUco ID', 'Left-Up X', 'Left-Up Y', 'Right-Up X', 'Right-Up Y', 'Right-Down X', 'Right-Down Y', 'Left-Down X', 'Left-Down Y', 'Distance', 'Yaw', 'Pitch', 'Roll'])
    csvwriter.writerows(results)

# Annotate video and save
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter('annotated_aruco_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    aruco_data = detect_aruco_codes(frame, aruco_dict, aruco_params)

    for aruco in aruco_data:
        pts = aruco['2d_points']
        pts = pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, str(aruco['id']), tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
