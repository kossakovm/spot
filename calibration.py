import cv2
import os
import numpy as np

def annotate_images(image_folder, window_name):
    """
    Process all images in image_folder.
    For each image, display it in a window (window_name) and wait for the user to click
    on the perceived center of the person. Returns a list of horizontal offsets.
    """
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    offsets = []  # List to store horizontal offsets (in pixels)

    for image_file in sorted(image_files):
        img = cv2.imread(image_file)
        if img is None:
            print(f"Could not load image: {image_file}")
            continue

        # Compute the nominal center x coordinate
        center_x = img.shape[1] / 2
        clicked_x = None

        def click_event(event, x, y, flags, param):
            nonlocal clicked_x
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_x = x
                print(f"Clicked x-coordinate on {os.path.basename(image_file)}: {x}")

        # Display image and set callback
        cv2.imshow(window_name, img)
        cv2.setMouseCallback(window_name, click_event)

        print(f"Annotate the center of the person in {os.path.basename(image_file)} and press any key when done.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

        if clicked_x is not None:
            offset = clicked_x - center_x
            offsets.append(offset)
            print(f"Image: {os.path.basename(image_file)} | Nominal center: {center_x:.2f} | Clicked: {clicked_x} | Offset: {offset:.2f} pixels\n")
        else:
            print(f"No click recorded for {os.path.basename(image_file)}. Skipping this image.\n")
    
    return offsets

# Define folders for left and right fisheye images
left_folder = "/Users/kossakovm/Desktop/thesis/spotproject/calibration_images_dual/calib_left"   # Folder with left fisheye images
right_folder = "/Users/kossakovm/Desktop/thesis/spotproject/calibration_images_dual/calib_right"  # Folder with right fisheye images

print("Annotating LEFT fisheye images...")
left_offsets = annotate_images(left_folder, "Left Camera")

print("Annotating RIGHT fisheye images...")
right_offsets = annotate_images(right_folder, "Right Camera")

# Compute average offsets if available
if left_offsets:
    avg_left = np.mean(left_offsets)
    print(f"\nAverage horizontal offset for LEFT fisheye images: {avg_left:.2f} pixels")
else:
    print("No valid annotations for LEFT fisheye images.")

if right_offsets:
    avg_right = np.mean(right_offsets)
    print(f"Average horizontal offset for RIGHT fisheye images: {avg_right:.2f} pixels")
else:
    print("No valid annotations for RIGHT fisheye images.")
