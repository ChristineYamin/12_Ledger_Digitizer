import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Reads an image and prepares it for OCR 
    by grayscaling and applying adaptive thresholding.    
    """
    # 1. Read the image
    img = cv2.imread(image_path)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)

    # 3. Apply adaptive thresholding to get rid of shadows 
    # This makes the paper white and the ink black
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed

# Test it
if __name__ == "__main__":
    # Put a test image to verify this work
    test_img = preprocess_image("")
    cv2.imwrite(", test_img")
    print("Pre-processing complete!")
