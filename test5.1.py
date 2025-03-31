import sys
import time
import math
import bosdyn.client
import bosdyn.client.util
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand, blocking_sit)
from bosdyn.client.robot_state import RobotStateClient

# Spot Connection Details
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

# Initialize SDK and Authenticate
sdk = bosdyn.client.create_standard_sdk('SpotCommandControl')
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
print("🔐 Authentication Successful!")

# Check if Spot is estopped
if robot.is_estopped():
    print("❌ Spot is Estopped! Release E-Stop and try again.")
    sys.exit(1)

# Acquire lease
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease = lease_client.acquire()
lease_keep_alive = LeaseKeepAlive(lease_client)

# Create command clients
command_client = robot.ensure_client(RobotCommandClient.default_service_name)
state_client = robot.ensure_client(RobotStateClient.default_service_name)

# Power on and stand
robot.power_on()
blocking_stand(command_client)
print("✅ Spot is standing and waiting for commands.")

def move_spot(direction, distance=1.0):
    """Moves Spot using a trajectory-based command."""
    transforms = state_client.get_robot_state().kinematic_state.transforms_snapshot
    dx, dy, dyaw = 0, 0, 0
    
    if direction == "forward":
        dx = distance
    elif direction == "backward":
        dx = -distance
    elif direction == "left":
        dy = distance
    elif direction == "right":
        dy = -distance
    elif direction == "rotate_left":
        dyaw = math.radians(45)
    elif direction == "rotate_right":
        dyaw = math.radians(-45)
    else:
        print("⚠️ Invalid direction!")
        return False
    
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal
    
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME, params=RobotCommandBuilder.mobility_params(stair_hint=False))
    
    print(f"🚀 Moving {direction} by {distance} meters...")
    cmd_id = command_client.robot_command(robot_cmd, end_time_secs=time.time() + 10)
    
    while True:
        feedback = command_client.robot_command_feedback(cmd_id)
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

while True:
    command = input("Enter command (e.g., '1 meter forward', '0.5 meter backward', '0.5 meter left', '0.5 meter right', '45 degree rotate_left', or 'q' to quit): ")
    
    if command.lower() == 'q':
        print("🛑 Command received: Sit down and stop.")
        blocking_sit(command_client)
        break
    
    try:
        tokens = command.split()
        if len(tokens) == 3 and tokens[1] == 'meter':
            distance = float(tokens[0])
            move_spot(tokens[2], distance)
        elif len(tokens) == 3 and tokens[1] == 'degree':
            angle = float(tokens[0])
            move_spot(tokens[2])
        else:
            print("⚠️ Invalid command format!")
    except ValueError:
        print("⚠️ Invalid numeric value. Use a valid number before 'meter' or 'degree'.")

print("🔻 Spot has stopped. Exiting script.")
