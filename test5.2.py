import sys
import time
import math
import threading
import bosdyn.client
import bosdyn.client.util
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME, get_se2_a_tform_b)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand, blocking_sit)
from bosdyn.client.robot_state import RobotStateClient
from pynput import keyboard

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
time.sleep(3)
blocking_stand(command_client)
print("✅ Spot is standing and waiting for commands.")

# Movement control flags
movement_direction = None
stop_movement = threading.Event()

def move_spot(direction, distance=0.4):
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
        dyaw = math.radians(15)
    elif direction == "rotate_right":
        dyaw = math.radians(-15)
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
    command_client.robot_command(robot_cmd, end_time_secs=time.time() + 10)

def on_press(key):
    global movement_direction, stop_movement
    try:
        key = key.char.lower()
        if key == 'w':
            movement_direction = "forward"
        elif key == 's':
            movement_direction = "backward"
        elif key == 'a':
            movement_direction = "left"
        elif key == 'd':
            movement_direction = "right"
        elif key == 'l':
            movement_direction = "rotate_left"
        elif key == 'r':
            movement_direction = "rotate_right"
        elif key == 'q':
            print("🛑 Quitting, making Spot sit...")
            blocking_sit(command_client)
            sys.exit(0)
        else:
            return
        
        stop_movement.clear()
        threading.Thread(target=move_spot, args=(movement_direction,), daemon=True).start()
    except AttributeError:
        pass

def on_release(key):
    global stop_movement
    stop_movement.set()

print("🎮 Use 'WASD' for movement, 'L' to rotate left, 'R' to rotate right, 'Q' to quit and sit. Press ESC to exit.")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

print("🔻 Spot has stopped. Exiting script.")
