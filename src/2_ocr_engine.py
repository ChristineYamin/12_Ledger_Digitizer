import easyocr
import csv
from paddleocr import PaddleOCR
import os

def extract_text(image_path):
    # Initialize the reader (only need to do this once)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Run OCR on the cleaned image
    print(f"Reading text from {image_path}")
    result = ocr.ocr(image_path, cls=True)
    extracted_data = []
    
    # Paddle OCR returns a nested list structure
    for line in result[0]:
        text = line[1][0]
        prob = line[1][1]
        # Only care about high-confidence results
        if prob > 0.4:
            extracted_data.append({'text': text, 'confidence': prob})
            print(f"Detected: {text} (Confidence: {prob:.2f})")
    return extracted_data

def save_to_csv(data, output_file="data/ledger_results.csv"):
    # Define the headers
    headers = ['Detected_Text', 'Confidence_Score']

    # Save to File
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for item in data:
            writer.writerow([item['text'], f"{item['confidence']:.2f}"])
    print(f"\nSuccessfully saved results to: {output_file}")

if __name__ == "__main__":
    # Ensure the path matches where the cleaned image is saved
    results = extract_text("data/cleaned_ledger.png")
    print("\n--- Processing Complete ---")
    print(f"Total valid items extraced: {len(results)}")
    if results:
        save_to_csv(results)
    else:
        print("No valud data found to save.")