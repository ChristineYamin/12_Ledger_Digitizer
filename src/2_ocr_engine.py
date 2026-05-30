import easyocr
import csv
import os

def extract_text(image_path):
    # Initialize the reader (only need to do this once)
    reader = easyocr.Reader(['en'], gpu=False)

    # Run OCR on the cleaned image
    print(f"Reading text from {image_path}")
    results = reader.readtext(image_path)
    extracted_data = []

    # Process results
    for (bbox, text, prob) in results:
        # Only care about high-confidence results
        if prob > 0.4:
            # Check if the text contains digits (is a number)
            if any(char.isdigit() for char in text):
                extracted_data.append({'text': text, "confidence": prob})
                print(f"Number Detected: {text} (Confidence: {prob:.2f})")

            else:
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