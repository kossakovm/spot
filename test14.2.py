import numpy as np
import cv2
import bosdyn.client
import bosdyn.client.util
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from ultralytics import YOLO  # YOLOv8
import time
import sys
import signal
import select
import termios
import tty
import math
from bosdyn.api import geometry_pb2

# ----------------- Spot Connection Details -----------------
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# ----------------- Initialize SDK and Authenticate -----------------
sdk = bosdyn.client.create_standard_sdk('SpotCombinedControl')
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
    print("✅ Spot is powered on!")
else:
    print("✅ Spot is already powered on.")

blocking_stand(robot_command_client)
print("📢 Spot is now standing!")

# ----------------- Load YOLO Model -----------------
model = YOLO('yolov8s.pt')
print("🎯 YOLOv8s Model Loaded!")

# ----------------- Calibration Biases & Movement Parameters -----------------
# (When combining images, you may need to tune this bias for the composite image.)
combined_horizontal_bias = 0  
# Movement control parameters
target_distance = 0.5         # Desired distance to human (meters)
distance_threshold = 0.05     # Deadband threshold (meters)
Kp = 0.5                      # Proportional gain for movement distance
max_move = 0.5                # Maximum allowed movement per command (meters)

# ----------------- Helper Functions -----------------
def get_spot_images():
    """
    Captures images from Spot’s left and right fisheye cameras and depth image.
    Returns:
      left_frame: RGB image from "frontleft_fisheye_image".
      right_frame: RGB image from "frontright_fisheye_image".
      depth_frame: Depth image from "frontleft_depth_in_visual_frame".
    """
    image_requests = [
        image_pb2.ImageRequest(image_source_name="frontleft_fisheye_image", quality_percent=50),
        image_pb2.ImageRequest(image_source_name="frontright_fisheye_image", quality_percent=50),
        image_pb2.ImageRequest(image_source_name="frontleft_depth_in_visual_frame", pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16)
    ]
    
    try:
        response = image_client.get_image(image_requests)
    except Exception as e:
        print("❌ Failed to capture images:", e)
        return None, None, None
    
    if len(response) < 3:
        print("❌ Not enough images received.")
        return None, None, None

    # Decode left fisheye image
    left_data = response[0].shot.image.data
    left_np = np.frombuffer(left_data, dtype=np.uint8)
    left_frame = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
    
    # Decode right fisheye image
    right_data = response[1].shot.image.data
    right_np = np.frombuffer(right_data, dtype=np.uint8)
    right_frame = cv2.imdecode(right_np, cv2.IMREAD_COLOR)
    
    # Process depth image; assume depth image shape matches left image.
    depth_img = response[2].shot.image
    if left_frame is not None:
        depth_frame = np.frombuffer(depth_img.data, dtype=np.uint16).reshape(left_frame.shape[:2])
    else:
        depth_frame = None

    if left_frame is None or right_frame is None or depth_frame is None:
        print("🚨 Failed to decode one or more images.")
        return None, None, None

    return left_frame, right_frame, depth_frame

def get_best_detection(detections, image_width, horizontal_bias):
    """
    Given a list of detections (from the combined image), return the detection with the smallest
    absolute offset from the adjusted image center.
    """
    best_det = None
    best_offset = float('inf')
    # The effective center of the combined image (plus any calibration bias)
    img_center = image_width / 2 + horizontal_bias

    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, class_id = det[:6]
            if int(class_id) == 0:  # Person class
                bbox_center = (x1 + x2) / 2
                offset = bbox_center - img_center
                if abs(offset) < abs(best_offset):
                    best_offset = offset
                    best_det = det
    return best_det, best_offset

def estimate_distance(bbox, depth_frame):
    """
    Estimate the distance to a person using depth data.
    bbox: [x1, y1, x2, y2] in left camera coordinates.
    """
    x1, y1, x2, y2 = bbox
    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    depth_values = depth_frame[
        max(0, center_y - 5): min(depth_frame.shape[0], center_y + 5),
        max(0, center_x - 5): min(depth_frame.shape[1], center_x + 5)
    ]
    
    valid_depths = depth_values[depth_values > 0]
    if valid_depths.size > 0:
        distance = np.median(valid_depths) / 1000.0  # mm to meters
        return distance
    else:
        print("⚠️ No valid depth data. Using fallback estimation.")
        return (depth_frame.shape[1] / (x2 - x1)) * 1.5

def rotate_spot(angle):
    """
    Rotates Spot in place by the specified angle (radians).
    Positive angles turn left; negative turn right.
    """
    print(f"🔄 Rotating Spot by {angle:.2f} radians...")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    current_pose = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    new_heading = current_pose.angle + angle

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=current_pose.x,
        goal_y=current_pose.y,
        goal_heading=new_heading,
        frame_name=ODOM_FRAME_NAME,
        params=RobotCommandBuilder.mobility_params(stair_hint=False)
    )

    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    time.sleep(0.1)
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)

    for _ in range(2):
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status == RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("✅ Rotation processing.")
        elif mobility_feedback.se2_trajectory_feedback.status == mobility_feedback.se2_trajectory_feedback.STATUS_AT_GOAL:
            print("✅ Rotation complete.")
            break
        else:
            print("⚠️ Rotation failed.")
            break
        time.sleep(0.5)

def move_spot_forward(distance=0.2):
    """Moves Spot forward by the given distance (meters)."""
    print(f"🔄 Moving forward {distance:.2f} meters...")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    body_tform_goal = math_helpers.SE2Pose(x=distance, y=0, angle=0)
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x,
        goal_y=out_tform_goal.y,
        goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME,
        params=RobotCommandBuilder.mobility_params(stair_hint=False)
    )
    
    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    time.sleep(0.1)
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)
    
    for _ in range(2):
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status == RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("✅ Forward movement processing.")
        elif mobility_feedback.se2_trajectory_feedback.status == mobility_feedback.se2_trajectory_feedback.STATUS_AT_GOAL:
            print("✅ Reached forward goal.")
            break
        else:
            print("⚠️ Forward movement failed.")
            break
        time.sleep(0.5)

def move_spot_backward(distance=0.2):
    """Moves Spot backward by the given distance (meters)."""
    print(f"🔄 Moving backward {distance:.2f} meters...")
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    body_tform_goal = math_helpers.SE2Pose(x=-distance, y=0, angle=0)
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x,
        goal_y=out_tform_goal.y,
        goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME,
        params=RobotCommandBuilder.mobility_params(stair_hint=False)
    )
    
    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    time.sleep(0.1)
    cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 1)
    
    for _ in range(2):
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status == RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print("✅ Backward movement processing.")
        elif mobility_feedback.se2_trajectory_feedback.status == mobility_feedback.se2_trajectory_feedback.STATUS_AT_GOAL:
            print("✅ Reached backward goal.")
            break
        else:
            print("⚠️ Backward movement failed.")
            break
        time.sleep(0.5)

def quat_from_euler(roll, pitch, yaw):
    """Convert Euler angles to a quaternion."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return math_helpers.Quat(w=w, x=x, y=y, z=z)

if not hasattr(math_helpers.Quat, 'to_quaternion'):
    def to_quaternion(self):
        return geometry_pb2.Quaternion(w=self.w, x=self.x, y=self.y, z=self.z)
    math_helpers.Quat.to_quaternion = to_quaternion

def adjust_spot_posture(height=0.1, pitch=0.2):
    """Adjust Spot's body posture (height and pitch)."""
    print(f"🔄 Adjusting posture: height={height:.2f} m, pitch={pitch:.2f} rad...")
    footprint_R_body = quat_from_euler(roll=0.0, pitch=pitch, yaw=0.0)
    robot_cmd = RobotCommandBuilder.synchro_stand_command(
        body_height=height,
        footprint_R_body=footprint_R_body
    )
    robot_command_client.robot_command(robot_cmd)

def cleanup_and_exit():
    """Stops Spot, releases lease, and exits cleanly."""
    print("🛑 Cleaning up... Spot sitting and releasing lease.")
    try:
        robot_command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Error while stopping Spot: {e}")
    lease_keep_alive.shutdown()
    sys.exit(0)

def is_key_pressed(key):
    """Check if a key is pressed (non-blocking)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ----------------- Main Loop -----------------
print("🔄 Starting combined control loop (using combined front image)...")
while True:
    # Exit if 'q' is pressed.
    if is_key_pressed('q'):
        print("🛑 'q' pressed. Exiting...")
        cleanup_and_exit()
        break

    # Capture images.
    left_frame, right_frame, depth_frame = get_spot_images()
    if left_frame is None or right_frame is None or depth_frame is None:
        print("❌ Image capture failed. Retrying...")
        time.sleep(0.2)
        continue

    # Combine the left and right fisheye images horizontally.
    try:
        combined_frame = cv2.hconcat([left_frame, right_frame])
    except Exception as e:
        print("⚠️ Failed to combine images:", e)
        time.sleep(0.2)
        continue

    # Run YOLO detection once on the combined image.
    results = model.predict(combined_frame, imgsz=640, conf=0.3)
    if not results or len(results[0].boxes) == 0:
        print("⚠️ No detections found in combined image.")
        time.sleep(0.2)
        continue

    detections = results[0].boxes.data.cpu().numpy()
    print(f"✅ Combined image: {len(detections)} objects detected.")

    # Use the combined image for rotation: find detection with minimal offset from center.
    combined_width = combined_frame.shape[1]
    best_det, offset = get_best_detection(detections, combined_width, combined_horizontal_bias)
    if best_det is not None:
        print(f"Detection offset from center: {offset:.2f} pixels")
    else:
        print("⚠️ No person detected for rotation.")

    # If offset exceeds a threshold, rotate.
    rotation_threshold_pixels = 10  # pixels deadband
    if best_det is not None and abs(offset) > rotation_threshold_pixels:
        max_rotation_angle = 0.785  # 15 degrees (in radians)
        rotation_angle = -max_rotation_angle * (offset / (combined_width / 2))
        print(f"Rotating: offset = {offset:.2f} px, angle = {rotation_angle:.4f} rad")
        rotate_spot(rotation_angle)
    else:
        print("✅ Person is horizontally centered; no rotation needed.")

    # ---- Movement Control using the detection from the combined image ----
    # For movement, we use the bounding box of the best detection.
    if best_det is not None:
        x1, y1, x2, y2, conf, class_id = best_det[:6]
        # Map bounding box to left camera coordinates.
        # If detection is in the right half, clip it to the left image dimensions.
        left_width = left_frame.shape[1]
        effective_x1 = x1 if x1 < left_width else left_width - 1
        effective_x2 = x2 if x2 < left_width else left_width - 1
        effective_bbox = [effective_x1, y1, effective_x2, y2]
        distance = estimate_distance(effective_bbox, depth_frame)
        print(f"Movement control: estimated distance = {distance:.2f} m")
        
        # Proportional control: move by a fraction of the error.
        error = distance - target_distance
        if abs(error) > distance_threshold:
            move_distance = Kp * error
            if move_distance > max_move:
                move_distance = max_move
            elif move_distance < -max_move:
                move_distance = -max_move
            
            if move_distance > 0:
                print(f"Person too far (error = {error:.2f} m). Moving forward {move_distance:.2f} m...")
                move_spot_forward(move_distance)
                adjust_spot_posture(height=0.0, pitch=0.0)
            else:
                print(f"Person too close (error = {error:.2f} m). Moving backward {abs(move_distance):.2f} m...")
                move_spot_backward(abs(move_distance))
                adjust_spot_posture(height=0.1, pitch=0.2)
        else:
            print(f"✅ Distance is optimal ({distance:.2f} m). No movement needed.")
            adjust_spot_posture(height=0.0, pitch=0.0)
    else:
        print("⚠️ No valid detection for movement control.")

    time.sleep(0.2)
