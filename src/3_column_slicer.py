import cv2
import numpy as np
import os

def slice_ledger_columns(image_path, output_dir="data/columns"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Canny to find edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use HoughLinesP to find vertical lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Extract X-coordinates of vertical lines
    x_coords = sorted(list(set([line[0][0] for line in lines if abs(line[0][0] - line[0][2]) < 5])))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Slice the image based on detected X-coordinates
    for i in range(len(x_coords) - 1):
        x1, x2 = x_coords[i], x_coords[i+1]
        col_img = img[:, x1:x2]
        cv2.imwrite(f"{output_dir}/col_{i}.png", col_img)
        print(f"Saved column {i} (X: {x1} to {x2})")

if __name__ == "__main__":
    slice_ledger_columns("data/cleaned_ledger.png")