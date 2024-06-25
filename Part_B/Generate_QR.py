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
