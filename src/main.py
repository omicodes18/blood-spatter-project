import cv2
import numpy as np
import os
import math


def read_input_image(images_folder: str, image_name: str = "blood_spatter.jpg") -> np.ndarray:
    
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
   
  
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = (5, 5)
    sigma = 0
    blurred = cv2.GaussianBlur(grayscale, kernel_size, sigma)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return grayscale, blurred, edges


def detect_droplet_contours(
    edges: np.ndarray, min_area: int = 100
) -> list[np.ndarray]:
   
   
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

   
    droplet_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    return droplet_contours


def extract_geometric_features(contour: np.ndarray) -> tuple[float, float, float]:
 
    area = cv2.contourArea(contour)

   
    rect = cv2.minAreaRect(contour)
    rect_width, rect_height = rect[1]
    
    width = min(rect_width, rect_height)
    length = max(rect_width, rect_height)

    return area, width, length


def estimate_angle_of_impact(width: float, length: float) -> float:
   
    if length <= 0 or width < 0:
        return 0.0
    ratio = width / length
    
    ratio = max(0.0, min(1.0, ratio))
    angle_rad = math.asin(ratio)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def main() -> None:
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    images_folder = os.path.join(project_root, "images")

   
    input_image_name = "blood_spatter.jpg"

  
    min_contour_area = 100

    
   
    image = read_input_image(images_folder, input_image_name)
    print("Input image loaded successfully.\n")

   
    grayscale, blurred, edges = preprocess_image(image)

    
    contours = detect_droplet_contours(edges, min_area=min_contour_area)
    print(f"Detected {len(contours)} droplet(s).\n")

    output_image = image.copy()

    print("-" * 60)
    print("Droplet Analysis Results")
    print("-" * 60)

    for idx, contour in enumerate(contours, start=1):
        area, width, length = extract_geometric_features(contour)
        angle_deg = estimate_angle_of_impact(width, length)

       
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

       
        print(f"Droplet ID/Index    : {idx}")
        print(f"Extracted Area     : {area:.2f} px²")
        print(f"Extracted Width    : {width:.2f} px")
        print(f"Extracted Length   : {length:.2f} px")
        print(f"Angle of Impact    : {angle_deg:.2f}°")
        print("-" * 60)

    window_name = "Blood Spatter Analysis - Processed Image (bounding boxes)"
    cv2.imshow(window_name, output_image)
    print("\nProcessed image displayed. Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
