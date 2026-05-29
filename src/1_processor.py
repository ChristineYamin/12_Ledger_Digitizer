import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """
    Reads an image and prepares it for OCR 
    by grayscaling and applying adaptive thresholding.    
    """
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image!")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)

    # 2. Apply Gaussian blur to reduce image noise
    # This helps the OCR engine focus on the shapes of the letters
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # 3. Apply adaptive thresholding to get rid of shadows 
    # This makes the paper white and the ink black
    processed = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed

# Test it
if __name__ == "__main__":
    # Ensure "data" directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # set the path to image path
    input_path = os.path.join("data", "handwritten ledgger.jpg")
    output_path = os.path.join("data", "cleaned_ledger.png")

    try:
        #Run the process
        cleaned_img = preprocess_image(input_path)

        # Save the result
        cv2.imwrite(output_path, cleaned_img)
        print(f"Success! Cleaned image saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

