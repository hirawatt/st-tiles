import cv2
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_image_select import image_select

def detect_floor(image):
    """Detects the floor in an image file.

    Args:
        image: The image file to be processed.

    Returns:
        A boolean value indicating whether the floor was detected.
    """
    try:
        # Convert the image to grayscale.
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the grayscale image to binarize it.
        thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]

        # Find the contours in the thresholded image.
        contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour, which is likely to be the floor.
        largest_contour = max(contours, key=cv2.contourArea, default=0)

        # Check if the largest contour is large enough to be considered a floor.
        if cv2.contourArea(largest_contour) > 10000:
            return True
        else:
            return False
    except:
        st.error("Floor not detected. Select another image.")

if __name__ == "__main__":
    st.set_page_config(
        "Interio",
        "🏠",
        initial_sidebar_state="expanded",
        layout="wide",
    )

    col1, col2 = st.columns(2)
    uploaded_file = st.sidebar.file_uploader("Upload Floor Image", type=['png', 'jpg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        col1.image(image, channels="BGR")

        # Detect the floor in the image.
        is_floor_detected = detect_floor(image)
        col1.info("Is floor detected? {}".format(is_floor_detected))

    img_file_buffer = col1.camera_input("Click an image of floor")
    if img_file_buffer is not None:
        # To read image file buffer as a PIL Image:
        img = Image.open(img_file_buffer)

        # To convert PIL Image to numpy array:
        img_array = np.array(img)

        # Check the shape of img_array:
        # Should output shape: (height, width, channels)
        st.write(img_array.shape)

        # Detect the floor in the image.
        is_floor_detected = detect_floor(img_array)
        st.info("Is floor detected? {}".format(is_floor_detected))

    # Display images to select from
    pattern_selected = image_select("Select any one", ["images/marble1.jpg", "images/marble2.jpg", "images/marble3.jpg", "images/marble4.jpg", "images/marble5.jpg"])
    #st.write(pattern_selected)
