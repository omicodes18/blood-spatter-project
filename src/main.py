"""
Blood Spatter Analysis System - Measurement-Based Analytical Model

This script implements a classical image processing pipeline for analyzing blood
droplet spatter patterns. It uses OpenCV for image processing and geometric
measurements to estimate the angle of impact for each droplet. No Machine
Learning, CNN, or Deep Learning is used—only measurement-based analytical models.

Suitable for academic Project-Based Learning (PBL) evaluation.
"""

import cv2
import numpy as np
import os
import math


def read_input_image(images_folder: str, image_name: str = "blood_spatter.jpg") -> np.ndarray:
    """
    Read the input image from the specified images folder.

    Args:
        images_folder: Path to the folder containing input images.
        image_name: Name of the input image file.

    Returns:
        The loaded image as a NumPy array (BGR format).

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    image_path = os.path.join(images_folder, image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Input image not found at '{image_path}'. "
            f"Please place your blood spatter image (e.g., blood_spatter.jpg) in the 'images/' folder."
        )
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from '{image_path}'. Check file format.")
    return image


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the input image for droplet contour detection.

    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur for denoising
    3. Apply Canny edge detection

    Args:
        image: Input BGR image.

    Returns:
        Tuple of (grayscale, blurred, edges) images.
    """
    # -------------------------------------------------------------------------
    # Step 1: Convert to Grayscale
    # -------------------------------------------------------------------------
    # Converting to grayscale reduces the image from 3 channels (BGR) to 1 channel,
    # which lowers computational complexity. It also focuses the analysis on
    # intensity (brightness) variations rather than color, which is beneficial for
    # edge detection and contour finding—since blood spatter edges are primarily
    # defined by intensity contrast against the background.
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------------------
    # Step 2: Apply Gaussian Blur
    # -------------------------------------------------------------------------
    # Gaussian blur smooths the image by averaging pixel values with their
    # neighbors using a Gaussian-weighted kernel. This reduces high-frequency
    # noise (e.g., sensor noise, small artifacts) that could produce false edges
    # in the next step. A smoothed image leads to more robust and continuous
    # edge detection results.
    kernel_size = (5, 5)
    sigma = 0
    blurred = cv2.GaussianBlur(grayscale, kernel_size, sigma)

    # -------------------------------------------------------------------------
    # Step 3: Canny Edge Detection
    # -------------------------------------------------------------------------
    # The Canny algorithm is chosen because it detects a wide range of edges
    # through gradient magnitude and direction, uses hysteresis thresholding to
    # suppress weak edges while keeping strong ones, and typically produces thin,
    # continuous edge contours. This is well-suited for identifying droplet
    # boundaries in blood spatter images.
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return grayscale, blurred, edges


def detect_droplet_contours(
    edges: np.ndarray, min_area: int = 100
) -> list[np.ndarray]:
    """
    Detect contours of blood droplets from the edge image.

    Contours are retrieved and filtered to exclude very small regions that are
    likely noise (e.g., dust, compression artifacts), keeping only contours
    that represent significant droplet shapes.

    Args:
        edges: Binary edge image from Canny detection.
        min_area: Minimum contour area in pixels; contours smaller than this
                  are ignored as noise.

    Returns:
        List of contours (each contour is a NumPy array of points).
    """
    # Find all contours in the binary edge image. We use RETR_EXTERNAL to get
    # only the outer boundaries of shapes (not nested contours inside droplets).
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out small contours that are likely noise—focus only on significant
    # droplet shapes. min_area acts as a simple threshold for "droplet-sized"
    # regions.
    droplet_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    return droplet_contours


def extract_geometric_features(contour: np.ndarray) -> tuple[float, float, float]:
    """
    Extract geometric features from a single droplet contour.

    The minimum bounding rectangle (minAreaRect) is used to obtain width and
    length. In bloodstain pattern analysis, the width is the dimension
    perpendicular to the direction of impact, and the length is the dimension
    parallel to it. These are crucial for the angle-of-impact formula.

    Args:
        contour: A single contour (array of points).

    Returns:
        Tuple of (area, width, length). Width and length are from the minimum
        bounding rectangle; the larger of the two is length, the smaller is width.
    """
    area = cv2.contourArea(contour)

    # cv2.minAreaRect returns (center, (width, height), angle). The width and
    # height of the minimum-area rectangle represent the dimensions perpendicular
    # and parallel to the impact direction. For the arcsin formula, we need
    # width = smaller dimension, length = larger dimension (so width/length <= 1).
    rect = cv2.minAreaRect(contour)
    rect_width, rect_height = rect[1]
    # Ensure width <= length so that width/length is in [0, 1] for arcsin.
    width = min(rect_width, rect_height)
    length = max(rect_width, rect_height)

    return area, width, length


def estimate_angle_of_impact(width: float, length: float) -> float:
    """
    Estimate the angle of impact (in degrees) using the analytical formula.

    Formula: Angle = arcsin(width / length)

    This derives from the assumption that a blood droplet striking a surface
    at an angle produces an ellipse whose aspect ratio (width/length) relates
    to the impact angle. When length is zero or width > length, the result
    is clamped to avoid invalid values.

    Args:
        width: Width of the droplet (min dimension of minimum bounding rect).
        length: Length of the droplet (max dimension of minimum bounding rect).

    Returns:
        Angle of impact in degrees [0, 90]. Returns 0 if length is 0 or too small.
    """
    if length <= 0 or width < 0:
        return 0.0
    ratio = width / length
    # Clamp ratio to [0, 1] to keep arcsin defined.
    ratio = max(0.0, min(1.0, ratio))
    angle_rad = math.asin(ratio)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def main() -> None:
    """Main entry point: load image, process, detect droplets, and report results."""
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Path to the images folder (one level up from 'src', then 'images').
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    images_folder = os.path.join(project_root, "images")

    # Input image name; change if using a different file.
    input_image_name = "blood_spatter.jpg"

    # Minimum contour area (pixels) to consider as a droplet; smaller contours
    # are treated as noise.
    min_contour_area = 100

    # -------------------------------------------------------------------------
    # Step 1: Input Image Reading
    # -------------------------------------------------------------------------
    image = read_input_image(images_folder, input_image_name)
    print("Input image loaded successfully.\n")

    # -------------------------------------------------------------------------
    # Step 2: Image Preprocessing
    # -------------------------------------------------------------------------
    grayscale, blurred, edges = preprocess_image(image)

    # -------------------------------------------------------------------------
    # Step 3: Droplet Contour Detection
    # -------------------------------------------------------------------------
    contours = detect_droplet_contours(edges, min_area=min_contour_area)
    print(f"Detected {len(contours)} droplet(s).\n")

    # -------------------------------------------------------------------------
    # Step 4 & 5: Geometric Feature Extraction and Angle of Impact Estimation
    # -------------------------------------------------------------------------
    # We will draw bounding boxes on a copy of the original image and print
    # results for each droplet.
    output_image = image.copy()

    print("-" * 60)
    print("Droplet Analysis Results")
    print("-" * 60)

    for idx, contour in enumerate(contours, start=1):
        area, width, length = extract_geometric_features(contour)
        angle_deg = estimate_angle_of_impact(width, length)

        # Draw minimum bounding rectangle (rotated box) as the bounding box.
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

        # Console output for each droplet.
        print(f"Droplet ID/Index    : {idx}")
        print(f"Extracted Area     : {area:.2f} px²")
        print(f"Extracted Width    : {width:.2f} px")
        print(f"Extracted Length   : {length:.2f} px")
        print(f"Angle of Impact    : {angle_deg:.2f}°")
        print("-" * 60)

    # -------------------------------------------------------------------------
    # Output: Display Processed Image with Bounding Boxes
    # -------------------------------------------------------------------------
    window_name = "Blood Spatter Analysis - Processed Image (bounding boxes)"
    cv2.imshow(window_name, output_image)
    print("\nProcessed image displayed. Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
