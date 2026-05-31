import cv2
import numpy as np
import os

def slice_ledger_columns(image_path, output_dir="data/columns"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
        
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Relaxed parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=int(height * 0.5), # 50% is safer
                            maxLineGap=20)
    
    # Safety Check: Did we actually find lines?
    if lines is None:
        print("No vertical lines detected! Try reducing the 'minLineLength' or 'threshold'.")
        return
    
    # Collect X-coords
    x_coords = sorted(list(set([line[0][0] for line in lines])))
    
    # Filter: Keep lines spaced at least 50 pixels apart
    filtered_x = [x_coords[0]]
    for x in x_coords[1:]:
        if x - filtered_x[-1] > 50:
            filtered_x.append(x)
            
    print(f"Detected {len(filtered_x)} columns.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(len(filtered_x) - 1):
        x1, x2 = filtered_x[i], filtered_x[i+1]
        col_img = img[:, x1:x2]
        cv2.imwrite(f"{output_dir}/column_{i}.png", col_img)
        print(f"Saved column {i} (X: {x1} to {x2})")

if __name__ == "__main__":
    slice_ledger_columns("data/cleaned_ledger.png")