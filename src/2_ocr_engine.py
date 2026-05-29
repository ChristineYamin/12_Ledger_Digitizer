import easyocr

def extract_text(image_path):
    # Initialize the reader (only need to do this once)
    reader = easyocr.Reader(['en'], gpu=True)

    # Run OCR on the cleaned image
    results = reader.readtext(image_path)

    extracted_data = []
    for (bbox, text, prob) in results:
        # Only care about high-confidence results
        if prob > 0.4:
            extracted_data.append(text)
            print(f"Detected: {text} (Confidence: {prob:.2f})")

    return extracted_data

if __name__ == "__main__":
    text = extract_text("data/scleaned_ledger.png")