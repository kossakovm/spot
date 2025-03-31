import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient

# Spot's connection details
SPOT_IP = "192.168.80.3"  # Change to your Spot's IP
USERNAME = "user"  # Change if needed
PASSWORD = "SPOT=LAB_in_K14"  # Change to your Spot's password

# Create SDK and Robot client
sdk = bosdyn.client.create_standard_sdk('SpotClient')
robot = sdk.create_robot(SPOT_IP)

try:
    # Authenticate with stored credentials
    robot.authenticate(USERNAME, PASSWORD)

    # Ensure Spot is time-synced
    robot.time_sync.wait_for_sync()

    # Get Spot's state
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_state = robot_state_client.get_robot_state()

    # Print connection confirmation
    if robot_state:
        print("✅ Spot is connected and ready!")

except Exception as e:
    print(f"⚠️ Error: {e}")

finally:
    # Stop time sync to prevent gRPC warnings
    robot.time_sync.stop()
    robot.logger.info("Shutting down Spot SDK.")
    print("🔄 SDK shutdown complete!")
