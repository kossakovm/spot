import os
import time
import numpy as np
import cv2

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2

# --- Spot Connection Details ---
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# --- Initialize SDK and Authenticate ---
sdk = bosdyn.client.create_standard_sdk('SpotDualFisheyeCalibration')
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

# --- Check if Spot is Estopped ---
if robot.is_estopped():
    print("❌ Spot is Estopped! Release E-Stop and try again.")
    exit(1)

# --- Acquire Lease & Keep Alive ---
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease_keep_alive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)

# --- Sync Time ---
robot.time_sync.wait_for_sync()

# --- Power On if Needed ---
if not robot.is_powered_on():
    print("🔌 Powering on Spot...")
    robot.power_on()
    # Give Spot time to fully power on.
    time.sleep(5)
    print("✅ Spot is now powered on!")
else:
    print("✅ Spot is already powered on.")

# --- Command Spot to Stand ---
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
blocking_stand(robot_command_client)
print("📢 Spot is now standing!")

# --- Create Clients ---
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)

# --- Directory to Save Calibration Images ---
save_dir = "calibration_images_dual"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for calibration images:", save_dir)

def get_dual_fisheye_images():
    """
    Captures images from Spot's front-left and front-right fisheye cameras.
    Returns (left_frame, right_frame) if successful, or (None, None) on failure.
    """
    image_requests = [
        image_pb2.ImageRequest(image_source_name="frontleft_fisheye_image", quality_percent=50),
        image_pb2.ImageRequest(image_source_name="frontright_fisheye_image", quality_percent=50)
    ]
    
    try:
        response = image_client.get_image(image_requests)
    except Exception as e:
        print("❌ Failed to capture images:", e)
        return None, None
    
    if len(response) < 2:
        print("❌ Not enough images in the response.")
        return None, None

    # Decode front-left image
    left_data = response[0].shot.image.data
    left_np = np.frombuffer(left_data, dtype=np.uint8)
    left_frame = cv2.imdecode(left_np, cv2.IMREAD_COLOR)

    # Decode front-right image
    right_data = response[1].shot.image.data
    right_np = np.frombuffer(right_data, dtype=np.uint8)
    right_frame = cv2.imdecode(right_np, cv2.IMREAD_COLOR)

    if left_frame is None or right_frame is None:
        print("🚨 Failed to decode one or both images.")
        return None, None

    return left_frame, right_frame

print("🔄 Starting dual-fisheye calibration image capture.")
print("Press 's' to save the current pair of images, or 'q' to quit.")

while True:
    left_frame, right_frame = get_dual_fisheye_images()
    if left_frame is None or right_frame is None:
        time.sleep(0.5)
        continue

    # Create a combined view to see both frames side by side
    # (Ensure they have the same height before stacking horizontally)
    height = min(left_frame.shape[0], right_frame.shape[0])
    if left_frame.shape[0] != right_frame.shape[0]:
        # Resize right_frame to match left_frame height (simple approach)
        right_frame = cv2.resize(right_frame, (int(right_frame.shape[1] * height / right_frame.shape[0]), height))
        left_frame = cv2.resize(left_frame, (int(left_frame.shape[1] * height / left_frame.shape[0]), height))

    combined_frame = np.hstack((left_frame, right_frame))
    cv2.imshow("Dual Fisheye Calibration", combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save each camera's image with a common timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        left_filename = os.path.join(save_dir, f"calib_left_{timestamp}.png")
        right_filename = os.path.join(save_dir, f"calib_right_{timestamp}.png")
        cv2.imwrite(left_filename, left_frame)
        cv2.imwrite(right_filename, right_frame)
        print(f"Saved left image: {left_filename}")
        print(f"Saved right image: {right_filename}")
    elif key == ord('q'):
        print("Exiting calibration image capture.")
        break

cv2.destroyAllWindows()
