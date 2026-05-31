import pytesseract
import csv
import cv2
import os

# POint this to where you installed Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Tesseract works best with RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
    # Run OCR on the cleaned image
    print(f"Reading text from {image_path}...")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    extracted_data = []
    # Iterate through results
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 40: # Confidence threshold
            text = data['text'][i].strip()
            if text:
                extracted_data.append({
                    'text': text,
                    'confidence': data['conf'][i] / 100,
                    'x': data['left'][i]
                })
                print(f"Detected: {text} at X={data['left'][i]} (Confidence: {data['conf'][i]/100:.2f})")
    
    return extracted_data

def save_to_csv(data, output_file="data/ledger_results.csv"):
    # Define the headers
    headers = ['Detected_Text', 'Confidence_Score', 'X_Position']

    # Save to File
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for item in data:
            writer.writerow([item['text'], f"{item['confidence']:.2f}", f"{item['x']:.0f}"])
    print(f"\nSuccessfully saved results to: {output_file}")

if __name__ == "__main__":
    results = extract_text("data/cleaned_ledger.png")
    if results:
        save_to_csv(results)
    else:
        print("No valid data found.")