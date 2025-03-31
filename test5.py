import sys
import time
import math
import bosdyn.client
import bosdyn.client.util
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

# Spot Connection Details
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# **1️⃣ Initialize SDK and Authenticate**
sdk = bosdyn.client.create_standard_sdk('SpotTrajectoryController')
robot = sdk.create_robot(SPOT_IP)

# ✅ **Fix Authentication Asking for Credentials Again**
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

# **2️⃣ Check if Spot is Estopped**
if robot.is_estopped():
    print("❌ Spot is Estopped! Release E-Stop and try again.")
    sys.exit(1)

# **3️⃣ Setup Clients**
lease_client = robot.ensure_client(LeaseClient.default_service_name)
robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)

# **4️⃣ Acquire Lease & Power On**
with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
    robot.time_sync.wait_for_sync()
    robot.power_on()
    print("✅ Spot is powered on!")

    # **5️⃣ Make Spot Stand**
    blocking_stand(robot_command_client)
    print("📢 Spot is now standing!")

    # **6️⃣ Define `move_spot` BEFORE Calling It**
    def move_spot(direction, robot_command_client, robot_state_client, distance=1.0):
        """Moves Spot in a specified direction using frame trajectory commands."""
        transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        # **9️⃣ Define Movement Directions**
        dx, dy, dyaw = 0, 0, 0  # Default: No Movement
        if direction == "forward":
            dx = distance  # Move Forward
        elif direction == "backward":
            dx = -distance  # Move Backward
        elif direction == "left":
            dy = distance  # Move Left
        elif direction == "right":
            dy = -distance  # Move Right
        elif direction == "rotate_left":
            dyaw = math.radians(45)  # Rotate Left 90°
        elif direction == "rotate_right":
            dyaw = math.radians(-45)  # Rotate Right 90°

        # **🔹 Transform Movement Goal into ODOM Frame**
        body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
        out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal

        # **🔹 Send Movement Command**
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME, params=RobotCommandBuilder.mobility_params(stair_hint=False))

        print(f"🚀 Moving {direction} by {distance} meters...")
        cmd_id = robot_command_client.robot_command(robot_cmd, end_time_secs=time.time() + 10)

        # **🔹 Monitor Movement Status**
        while True:
            feedback = robot_command_client.robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print("⚠️ Failed to reach the goal.")
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print("✅ Arrived at the goal.")
                return True
            time.sleep(1)
        return True

    # **7️⃣ Read Movement Command from CLI AFTER Function Definition**
    if len(sys.argv) > 1:
        direction = sys.argv[1].lower()
        if direction in ["forward", "backward", "left", "right", "rotate_left", "rotate_right"]:
            move_spot(direction, robot_command_client, robot_state_client)
        else:
            print("⚠️ Invalid command! Use: forward, backward, left, right, rotate_left, rotate_right.")
    else:
        print("ℹ️ Usage: python move_spot.py <direction>")

    # **8️⃣ Stop Spot After Movement**
    robot_command_client.robot_command(RobotCommandBuilder.stop_command())
    print("🛑 Spot has stopped.")
