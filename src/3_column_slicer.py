import cv2
import numpy as np
import os

def slice_ledger_columns(image_path, output_dir="data/columns"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Binarize to find where the content is
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Sum the pixels vertically. Columns with no text will have near-zero values.
    column_sums = np.sum(binary, axis=0)
    
    # 3. Find gaps (where column_sums are near zero)
    gaps = np.where(column_sums < 500)[0] # Threshold for "empty" space
    
    # 4. Identify column breaks
    breaks = [0]
    for i in range(1, len(gaps)):
        if gaps[i] - gaps[i-1] > 50: # Minimum column width
            breaks.append(gaps[i])
    breaks.append(img.shape[1])
    
    # 5. Save the chunks
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for i in range(len(breaks) - 1):
        col_img = img[:, breaks[i]:breaks[i+1]]
        # Only save columns that aren't just empty noise
        if col_img.shape[1] > 20:
            cv2.imwrite(f"{output_dir}/col_{i}.png", col_img)
            print(f"Saved col_{i}.png (Width: {col_img.shape[1]})")

if __name__ == "__main__":
    slice_ledger_columns("data/cleaned_ledger.png")