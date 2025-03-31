import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
import cv2
import numpy as np

def get_spot_camera_feed(robot_ip):
    """Connect to Spot and fetch camera frames."""
    sdk = bosdyn.client.create_standard_sdk('SpotCameraClient')
    robot = sdk.create_robot(robot_ip)
    bosdyn.client.util.authenticate(robot)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    sources = ['frontleft_fisheye_image', 'frontright_fisheye_image']
    while True:
        # Fetch images
        image_responses = image_client.get_image_from_sources(sources)
        for img_resp in image_responses:
            img = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            cv2.imshow(img_resp.source.name, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Run with Spot IP
get_spot_camera_feed('192.168.80.3')  # Change to your Spot's IP
