"""
Flask API for fake news classification
Run with: python app.py
Then make requests to: http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
import io

import os

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import pytesseract

    # ── Explicitly tell pytesseract where tesseract.exe lives on Windows ──
    # The binary is installed but may not be on PATH when Flask runs.
    _TESS_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
    ]
    for _p in _TESS_PATHS:
        if os.path.isfile(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

    OCR_AVAILABLE = True
    print(f"Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract or Pillow not installed. /predict-image endpoint will be unavailable.")
    print("Install with: pip install pytesseract Pillow")
    print("Also ensure Tesseract OCR is installed: https://github.com/UB-Mannheim/tesseract/wiki")


app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Load model and vectorizer at startup
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found!")
    print("Please run model_trainer.py first to train and save the model.")
    exit(1)

def wordopt(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def predict_fake_news(text):
    """Predict if text is fake news"""
    # Preprocess
    cleaned_text = wordopt(text)
    
    # Vectorize
    text_vector = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    # Format response
    result = {
        "prediction": "Fake News" if prediction == 0 else "Not Fake News",
        "confidence": float(max(probability)) * 100,
        "probabilities": {
            "fake_news": float(probability[0]) * 100,
            "not_fake_news": float(probability[1]) * 100
        }
    }
    
    return result

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for fake news prediction
    
    Expected JSON input:
    {
        "text": "Your news text here"
    }
    
    Example response:
    {
        "prediction": "Fake News",
        "confidence": 85.23,
        "probabilities": {
            "fake_news": 85.23,
            "not_fake_news": 14.77
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in JSON input"
            }), 400
        
        news_text = data.get('text', '').strip()
        
        if not news_text:
            return jsonify({
                "error": "Text field cannot be empty"
            }), 400
        
        result = predict_fake_news(news_text)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/predict-image', methods=['POST'])
def predict_image():
    """
    API endpoint for image-based fake news prediction.
    Accepts a multipart/form-data image file, extracts text via OCR,
    then runs the fake news classifier on the extracted text.

    Expected: multipart/form-data with key 'image'
    """
    if not OCR_AVAILABLE:
        return jsonify({
            "error": "OCR dependencies not installed. Run: pip install pytesseract Pillow"
        }), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    try:
        # Open image with Pillow
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ── Pre-process for better OCR accuracy ──
        # 1. Upscale small images (Tesseract works best at ~300 DPI)
        w, h = image.size
        if max(w, h) < 1000:
            scale = 1000 / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # 2. Convert to grayscale
        gray = image.convert("L")

        # 3. Boost contrast so light-on-dark text becomes readable
        gray = ImageEnhance.Contrast(gray).enhance(2.0)

        # 4. Try three Tesseract page-segmentation modes; keep the longest result
        best_text = ""
        for psm in (6, 3, 11):
            cfg = f"--oem 3 --psm {psm}"
            t = pytesseract.image_to_string(gray, lang="eng", config=cfg).strip()
            if len(t) > len(best_text):
                best_text = t

        extracted_text = best_text

        if not extracted_text:
            return jsonify({
                "error": "No text could be extracted from the image. Please upload an image containing readable text.",
                "extracted_text": ""
            }), 422

        # Run the existing ML model on extracted text
        result = predict_fake_news(extracted_text)
        result["extracted_text"] = extracted_text
        result["ocr_char_count"] = len(extracted_text)
        return jsonify(result), 200


    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "OK",
        "model": "Fake News Classifier",
        "ocr_available": OCR_AVAILABLE,
        "version": "1.0"
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage instructions"""
    return jsonify({
        "message": "Fake News Classifier API",
        "endpoints": {
            "POST /predict": "Classify news as fake or real",
            "GET /health": "Health check",
            "GET /": "This help message"
        },
        "example_usage": {
            "url": "POST http://localhost:5000/predict",
            "body": {
                "text": "Your news text here"
            }
        }
    }), 200

if __name__ == '__main__':
    print("Starting Fake News Classifier API...")
    print("Visit http://localhost:5000 for API documentation")
    print("To make predictions, send POST requests to http://localhost:5000/predict")
    app.run(debug=True, host='0.0.0.0', port=5000)
