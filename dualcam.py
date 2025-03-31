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

# ----------------- Spot Connection Details -----------------
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# ----------------- Initialize SDK and Authenticate -----------------
sdk = bosdyn.client.create_standard_sdk('SpotPanoramicStitch')
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

if robot.is_estopped():
    print("❌ Spot is Estopped! Release E-Stop and try again.")
    exit(1)

# ----------------- Setup Clients -----------------
lease_client = robot.ensure_client(LeaseClient.default_service_name)
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
image_client = robot.ensure_client(ImageClient.default_service_name)

# Acquire lease & power on
try:
    lease_keep_alive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
except bosdyn.client.lease.ResourceAlreadyClaimedError:
    print("⚠️ Spot's lease is already held. Taking over...")
    lease_client.take()
    lease_keep_alive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)

robot.time_sync.wait_for_sync()

if not robot.is_powered_on():
    print("🔌 Powering on Spot...")
    robot.power_on()
    time.sleep(5)  # Allow time for Spot to fully power on.
    print("✅ Spot is now powered on!")
else:
    print("✅ Spot is already powered on.")

blocking_stand(robot_command_client)
print("📢 Spot is now standing!")

# ----------------- Directory for Saved Images -----------------
save_dir = "stitched_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created directory for saved images:", save_dir)

# ----------------- Function to Capture Fisheye Images -----------------
def get_fisheye_images():
    """
    Captures images from Spot's front-left and front-right fisheye cameras.
    Returns (left_frame, right_frame) if successful; otherwise (None, None).
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
        print("❌ Not enough images received.")
        return None, None

    # Decode left fisheye image
    left_data = response[0].shot.image.data
    left_np = np.frombuffer(left_data, dtype=np.uint8)
    left_frame = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
    
    # Decode right fisheye image
    right_data = response[1].shot.image.data
    right_np = np.frombuffer(right_data, dtype=np.uint8)
    right_frame = cv2.imdecode(right_np, cv2.IMREAD_COLOR)
    
    if left_frame is None or right_frame is None:
        print("🚨 Failed to decode one or both images.")
        return None, None

    return left_frame, right_frame

# ----------------- Main Loop -----------------
print("🔄 Starting panoramic view. Press 's' to save image, 'q' to quit.")
while True:
    left_frame, right_frame = get_fisheye_images()
    if left_frame is None or right_frame is None:
        time.sleep(0.5)
        continue

    # Optionally, ensure both images have the same height
    height = min(left_frame.shape[0], right_frame.shape[0])
    left_frame = cv2.resize(left_frame, (int(left_frame.shape[1] * height / left_frame.shape[0]), height))
    right_frame = cv2.resize(right_frame, (int(right_frame.shape[1] * height / right_frame.shape[0]), height))
    
    # Attempt stitching using OpenCV's stitcher.
    stitcher = cv2.Stitcher_create() if hasattr(cv2, 'Stitcher_create') else cv2.createStitcher()
    status, stitched = stitcher.stitch([left_frame, right_frame])
    
    if status != cv2.Stitcher_OK:
        print("⚠️ Stitching failed (status code: {}), using side-by-side concatenation as fallback.".format(status))
        stitched = cv2.hconcat([left_frame, right_frame])
    
    cv2.imshow("Stitched Panoramic View", stitched)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"stitched_{timestamp}.png")
        cv2.imwrite(filename, stitched)
        print("Saved stitched image to:", filename)
    elif key == ord('q'):
        print("Exiting...")
        break

cv2.destroyAllWindows()
