import cv2
import numpy as np
import time
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2

# ----------------- Spot Connection Details -----------------
SPOT_IP = "192.168.80.3"
USERNAME = "user"
PASSWORD = "SPOT=LAB_in_K14"

def undistort_and_cylindrical(img, camera="frontleft"):
    """
    A placeholder function to undistort a fisheye image and warp it to a cylindrical projection.
    Replace K and D with your actual calibration parameters.
    """
    h, w = img.shape[:2]

    # Example (placeholder) calibration parameters for each camera:
    if camera == "frontleft":
        # Intrinsic matrix (focal length and principal point)
        K = np.array([[300.0, 0, w/2],
                      [0, 300.0, h/2],
                      [0, 0, 1]], dtype=np.float32)
        # Distortion coefficients (placeholder values)
        D = np.array([[-0.1, 0.01, 0, 0]], dtype=np.float32)
    elif camera == "frontright":
        K = np.array([[300.0, 0, w/2],
                      [0, 300.0, h/2],
                      [0, 0, 1]], dtype=np.float32)
        D = np.array([[-0.1, 0.01, 0, 0]], dtype=np.float32)
    else:
        K = np.eye(3, dtype=np.float32)
        D = np.zeros((1,4), dtype=np.float32)

    # Undistort the fisheye image.
    # (For real use, consider cv2.fisheye.undistortImage with proper K and D.)
    undistorted = cv2.fisheye.undistortImage(img, K, D, Knew=K, new_size=(w, h))

    # Create a cylindrical projection.
    # For each pixel in the output, compute the corresponding source pixel.
    f = K[0, 0]  # approximate focal length
    cyl = np.zeros_like(undistorted)
    for y in range(h):
        for x in range(w):
            # Compute angle theta relative to the center
            theta = (x - w/2) / f
            # Compute the height in cylindrical coordinates (normalized)
            h_val = (y - h/2) / f
            # Map back to source coordinates:
            x_src = f * np.tan(theta) + w/2
            y_src = f * h_val / np.cos(theta) + h/2
            if 0 <= int(x_src) < w and 0 <= int(y_src) < h:
                cyl[y, x] = undistorted[int(y_src), int(x_src)]
    return cyl

def stitch_images(left_img, right_img):
    """
    Attempt to stitch the two warped images using OpenCV's Stitcher.
    If stitching fails, fall back to a simple horizontal concatenation.
    """
    stitcher = cv2.Stitcher_create() if hasattr(cv2, 'Stitcher_create') else cv2.createStitcher()
    status, pano = stitcher.stitch([left_img, right_img])
    if status == cv2.Stitcher_OK:
        return pano
    else:
        print("⚠️ Stitching failed (status code: {}), falling back to hconcat.".format(status))
        return cv2.hconcat([left_img, right_img])

def get_fisheye_images(image_client):
    """
    Retrieves Spot's front-left and front-right fisheye images.
    Returns (left_frame, right_frame) or (None, None) on failure.
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

    # Decode images from bytes
    left_np = np.frombuffer(response[0].shot.image.data, dtype=np.uint8)
    right_np = np.frombuffer(response[1].shot.image.data, dtype=np.uint8)
    left_frame = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
    right_frame = cv2.imdecode(right_np, cv2.IMREAD_COLOR)
    if left_frame is None or right_frame is None:
        print("🚨 Failed to decode images.")
        return None, None

    return left_frame, right_frame

def main():
    # Initialize SDK and authenticate with Spot.
    sdk = bosdyn.client.create_standard_sdk('StitchFrontImages')
    robot = sdk.create_robot(SPOT_IP)
    robot.authenticate(USERNAME, PASSWORD)
    if robot.is_estopped():
        print("❌ Spot is estopped. Please release E-Stop and try again.")
        return

    robot.time_sync.wait_for_sync()
    image_client = robot.ensure_client(ImageClient.default_service_name)

    print("Press 's' to save a stitched image, or 'q' to quit.")
    while True:
        left_frame, right_frame = get_fisheye_images(image_client)
        if left_frame is None or right_frame is None:
            time.sleep(0.5)
            continue

        # Optionally resize images for speed
        # For example: left_frame = cv2.resize(left_frame, (left_frame.shape[1]//2, left_frame.shape[0]//2))
        #              right_frame = cv2.resize(right_frame, (right_frame.shape[1]//2, right_frame.shape[0]//2))

        # Apply undistortion and cylindrical warp to each fisheye image.
        left_warped = undistort_and_cylindrical(left_frame, "frontleft")
        right_warped = undistort_and_cylindrical(right_frame, "frontright")

        # Stitch the two warped images into a single panoramic view.
        panorama = stitch_images(left_warped, right_warped)

        cv2.imshow("Stitched Panoramic View", panorama)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"stitched_{timestamp}.png"
            cv2.imwrite(filename, panorama)
            print("Saved stitched image as:", filename)
        elif key == ord('q'):
            print("Exiting.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
