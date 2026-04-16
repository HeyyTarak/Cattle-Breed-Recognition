from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io
import logging
from pathlib import Path
from PIL import Image
from flask_cors import CORS
from werkzeug.utils import secure_filename
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration with environment variables support
class Config:
    """Application configuration"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    MODEL_PATH = os.environ.get('MODEL_PATH', 'breed_classifier.h5')
    DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset')
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.45'))
    IMAGE_SIZE = (128, 128)
    PORT = int(os.environ.get('PORT', 5001))
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'

app.config.from_object(Config)

# Global variables
model: Optional[tf.keras.Model] = None
class_names: List[str] = []

# Comprehensive Livestock Database with enhanced structure
BREED_INFO: Dict[str, Dict[str, str]] = {
    "Ayrshire": {
        "Milk Production per day": "24.6 liters (4.13% butterfat)",
        "Height": "1.5 meters at shoulder",
        "Weight": "Up to 725 kg",
        "Types of diseases": "Bovine pneumonia, Johne's disease, Bloat, Foot rot, Pinkeye",
        "Price": "Approx. INR 60,000",
        "Origin Place": "Ayrshire, Scotland",
        "Life Span": "10-12 years (productive life)",
        "Types of food best for that breed": "Grass, hay, and high-quality forages",
        "Dung per day": "29.5 kg"
    },
    "Brown Swiss": {
        "Milk Production per day": "25-30 liters (High fat and protein content)",
        "Height": "1.42m - 1.54m at the shoulder",
        "Weight": "600-750 kg",
        "Types of diseases": "Hardy, but prone to Weaver, Spiderleg, and Spinal Dysmyelination",
        "Price": "INR 45,000 - 55,000 (Karan Swiss crossbreed)",
        "Origin Place": "Switzerland",
        "Life Span": "15-20 years (Known for longevity)",
        "Types of food best for that breed": "Balanced green/dry fodder, concentrate mix, and minerals",
        "Dung per day": "48 kg (for mature 636kg cow)"
    },
    "Gir": {
        "Milk Production per day": "6-10 liters (can reach 15-20 liters with high-end management)",
        "Height": "130 cm (Female) / 135 cm (Male)",
        "Weight": "385 kg (Female) / 545 kg (Male)",
        "Types of diseases": "Highly resistant to tropical diseases and ticks; prone to joint/respiratory issues",
        "Price": "INR 50,000 - 1,50,000 (Dependent on age and lactation)",
        "Origin Place": "Gir Forest, Gujarat, India",
        "Life Span": "12-15 years",
        "Types of food best for that breed": "Napier grass, Berseem, wheat straw, and salt mix",
        "Dung per day": "High volume; excellent for organic manure and biogas"
    },
    "Murrah": {
        "Milk Production per day": "8-16 liters (Lactation yield: 1,500 - 2,500 kg)",
        "Height": "132 cm (Female) / 142 cm (Male)",
        "Weight": "450-550 kg (Female) / 550-650 kg (Male)",
        "Types of diseases": "Resistant but susceptible to FMD, Mastitis, and respiratory infections",
        "Price": "INR 80,000 - 2,50,000 (Based on milk yield and physical build)",
        "Origin Place": "Haryana and Punjab, India",
        "Life Span": "Approx. 20 years",
        "Types of food best for that breed": "Protein-rich berseem, lucerne, paddy straw, and concentrate feed",
        "Dung per day": "Significant amount; high nitrogen content"
    }
}

def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_resources() -> bool:
    """Initialize the AI model and class labels on startup"""
    global model, class_names
    
    try:
        model_path = Path(app.config['MODEL_PATH'])
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Load model with custom objects if needed
        model = load_model(str(model_path))
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Load class names from dataset folder or use defaults
        dataset_path = Path(app.config['DATASET_PATH'])
        if dataset_path.exists() and dataset_path.is_dir():
            class_names = sorted([
                d.name for d in dataset_path.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ])
            logger.info(f"Loaded {len(class_names)} classes from dataset")
        else:
            class_names = ["Ayrshire", "Brown Swiss", "Gir", "Murrah"]
            logger.info("Using default class names")
        
        logger.info("AI Marketplace Engine successfully initialized")
        return True
        
    except Exception as e:
        logger.error(f"Initialization Error: {e}", exc_info=True)
        return False

def preprocess_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Preprocess image for model prediction"""
    try:
        # Open and convert image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize image
        img = img.resize(app.config['IMAGE_SIZE'])
        
        # Convert to array and normalize
        img_array = image.img_to_array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None

def predict_breed(img_array: np.ndarray) -> Tuple[str, float, bool]:
    """Make prediction on preprocessed image"""
    try:
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        
        # Determine if it's cattle based on confidence threshold
        is_cattle = confidence > app.config['CONFIDENCE_THRESHOLD']
        
        # Get breed name
        raw_name = class_names[predicted_idx]
        display_name = raw_name.replace('_', ' ').title()
        
        return display_name, confidence, is_cattle
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

@app.route('/')
def index() -> str:
    """Serve the marketplace interface"""
    try:
        html_path = Path('cattle_connect.html')
        if html_path.exists():
            return html_path.read_text(encoding='utf-8')
        else:
            logger.error("cattle_connect.html not found")
            return "Error: cattle_connect.html missing. Please ensure the file exists.", 404
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return "Internal server error", 500

@app.route('/health', methods=['GET'])
def health_check() -> Tuple[Dict, int]:
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'classes_loaded': len(class_names) > 0,
        'config': {
            'confidence_threshold': app.config['CONFIDENCE_THRESHOLD'],
            'image_size': app.config['IMAGE_SIZE']
        }
    }), 200

@app.route('/predict', methods=['POST'])
def predict_endpoint() -> Tuple[Dict, int]:
    """Process the image and return breed details + cattle status"""
    
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded")
        return jsonify({'error': 'Model not initialized. Please check server logs.'}), 503
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400
    
    try:
        # Read file content
        img_bytes = file.read()
        
        # Check file size
        if len(img_bytes) > app.config['MAX_FILE_SIZE']:
            return jsonify({'error': f'File too large. Maximum size: {app.config["MAX_FILE_SIZE"] // (1024*1024)}MB'}), 400
        
        # Preprocess image
        img_array = preprocess_image(img_bytes)
        if img_array is None:
            return jsonify({'error': 'Invalid image format or corrupted file'}), 400
        
        # Make prediction
        display_name, confidence, is_cattle = predict_breed(img_array)
        
        # Log prediction
        logger.info(f"Prediction: {display_name} (confidence: {confidence:.2f}, is_cattle: {is_cattle})")
        
        # Get breed info if it's cattle
        info = BREED_INFO.get(display_name, {}) if is_cattle else {}
        
        return jsonify({
            'breed': display_name if is_cattle else "Unknown Object",
            'confidence': round(confidence, 4),
            'isCattle': is_cattle,
            'info': info
        })
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}", exc_info=True)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error) -> Tuple[Dict, int]:
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error) -> Tuple[Dict, int]:
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error) -> Tuple[Dict, int]:
    """Handle file too large errors"""
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    # Load resources
    if not load_resources():
        logger.warning("Failed to load model. Predictions will not work.")
    
    # Run the application
    logger.info(f"Starting Flask server on port {app.config['PORT']}")
    app.run(
        debug=app.config['DEBUG'],
        port=app.config['PORT'],
        host='0.0.0.0',
        threaded=True  # Enable threading for better performance
    )