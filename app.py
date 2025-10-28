from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
import numpy as np

app = Flask(__name__)

class SentimentPredictor:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self.model_loaded = self.load_model()
    
    def load_model(self):
        try:
            model_path = 'models/sentiment_model.pkl'
            if not os.path.exists(model_path):
                print("❌ ERROR: Model file not found at models/sentiment_model.pkl")
                print("💡 Please make sure you downloaded the .pkl file from Google Colab")
                return False
                
            print("🔄 Loading trained ML model...")
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.tfidf = model_data['tfidf_vectorizer']
            self.label_encoder = model_data['label_encoder']
            
            print("✅ ML Model loaded successfully!")
            print(f"✅ Model type: {type(self.model).__name__}")
            print(f"✅ Available classes: {list(self.label_encoder.classes_)}")
            print(f"✅ TF-IDF features: {self.tfidf.get_feature_names_out().shape[0]}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, text):
        if not self.model_loaded:
            raise Exception("ML model not loaded. Please check if sentiment_model.pkl exists in models/ folder")
        
        # Clean the text (same preprocessing as during training)
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()
        
        if not cleaned_text:
            return "neutral", 50.0, {'positive': 33, 'negative': 33, 'neutral': 34}
            
        # Transform using the same TF-IDF vectorizer used in training
        features = self.tfidf.transform([cleaned_text])
        
        # Get prediction from trained model
        prediction = self.model.predict(features)[0]
        sentiment = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities for all classes from trained model
        probabilities = self.model.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        
        # Create probability distribution
        sentiment_probs = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            sentiment_probs[class_name] = round(probabilities[i] * 100, 2)
        
        print(f"🔍 Prediction: '{text[:50]}...' → {sentiment} (confidence: {confidence:.2f}%)")
        
        return sentiment, round(confidence, 2), sentiment_probs

predictor = SentimentPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {'positive': 0, 'negative': 0, 'neutral': 0}
            })
        
        # This will now throw an error if model is not loaded
        sentiment, confidence, probabilities = predictor.predict(text)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': probabilities,
            'text': text
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'sentiment': 'neutral',
            'confidence': 0.0,
            'probabilities': {'positive': 0, 'negative': 0, 'neutral': 0}
        })

if __name__ == '__main__':
    print("🚀 Starting Flask application...")
    print("📁 Current directory:", os.getcwd())
    
    # Check if required files exist
    required_files = [
        'models/sentiment_model.pkl',
        'templates/index.html',
        'static/styles.css'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
    
    if not os.path.exists('models/sentiment_model.pkl'):
        print("\n❌ CRITICAL: Model file not found!")
        print("💡 Please download 'sentiment_model.pkl' from Google Colab and place it in models/ folder")
        print("💡 The file should be in your downloads from the Colab training")
    
    app.run(debug=True, host='0.0.0.0', port=5000)