import os
import cv2
import numpy as np

def annotate_images(folder):
    """
    For each image in the given folder, displays the image and waits for the user to click
    on the perceived center of the person. Returns a list of horizontal offsets (in pixels)
    relative to the nominal center (image width/2).
    """
    # Get list of image files (png, jpg, jpeg)
    image_files = sorted([f for f in os.listdir(folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    offsets = []
    
    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue
        
        # Calculate nominal center of the image (in pixels)
        nominal_center = img.shape[1] / 2.0
        
        clicked_x = None  # This will store the x-coordinate of the user's click
        
        # Define a callback function to capture the mouse click
        def click_event(event, x, y, flags, param):
            nonlocal clicked_x
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_x = x
                print(f"Clicked x-coordinate on {image_file}: {x}")
        
        window_name = f"Annotate: {image_file}"
        cv2.imshow(window_name, img)
        cv2.setMouseCallback(window_name, click_event)
        
        print(f"Annotate the center of the person in {image_file} by clicking on it, then press any key.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        
        if clicked_x is not None:
            offset = clicked_x - nominal_center
            offsets.append(offset)
            print(f"{image_file} | Nominal center: {nominal_center:.2f} px | Clicked: {clicked_x} px | Offset: {offset:.2f} px\n")
        else:
            print(f"No click recorded for {image_file}. Skipping this image.\n")
            
    return offsets

if __name__ == "__main__":
    folder = input("Enter the folder path containing your combined images: ").strip()
    if not os.path.isdir(folder):
        print("Invalid folder path. Exiting.")
        exit(1)
    
    offsets = annotate_images(folder)
    
    if offsets:
        avg_offset = np.mean(offsets)
        print(f"\nAverage combined_horizontal_bias: {avg_offset:.2f} pixels")
    else:
        print("No valid annotations were collected.")
