import cv2
import numpy as np
import streamlit as st

def detect_floor(image):
    """Detects the floor in an image file.

    Args:
        image: The image file to be processed.

    Returns:
        A boolean value indicating whether the floor was detected.
    """

    # Convert the image to grayscale.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale image to binarize it.
    thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]

    # Find the contours in the thresholded image.
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, which is likely to be the floor.
    largest_contour = max(contours, key=cv2.contourArea)

    # Check if the largest contour is large enough to be considered a floor.
    if cv2.contourArea(largest_contour) > 10000:
        return True
    else:
        return False

if __name__ == "__main__":
    st.title("Marble/Tile Selection App")

    selection = st.radio("Upload file/Take live image?", ["Offline", "Live"])
    if selection == "Offline":
        st.file_uploader("Upload Floor Image")
    else:
        camera_image = st.camera_input("Click an image of floor")

    # Load the image file.
    image = cv2.imread("image.jpg")
    #image = cv2.imread(camera_image)

    # Detect the floor in the image.
    is_floor_detected = detect_floor(image)
    #is_floor_detected = detect_floor(camera_image)

    # Print the result.
    st.info("Is floor detected? {}".format(is_floor_detected))
