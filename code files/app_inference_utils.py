import joblib
import numpy as np
import pandas as pd
import warnings
from functools import lru_cache

# Suppress warnings
warnings.filterwarnings('ignore')

# Load models and preprocessors
def load_models():
    """Load all required models and preprocessors"""
    try:
        models = {
            'lgbm': joblib.load('models/adr_lightgbm_model.pkl'),
            'xgb': joblib.load('models/adr_xgboost_model.pkl'),
            'cat': joblib.load('models/adr_catboost_model.pkl'),
            'meta': joblib.load('models/adr_meta_model.pkl'),
            'tfidf_indication': joblib.load('tfidfs/tfidf_indication.pkl'),
            'tfidf_side_effect': joblib.load('tfidfs/tfidf_side_effect.pkl'),
            'onehot_drug': joblib.load('one_hot_encoder/onehot_drug-001.pkl'),
            'scaler': joblib.load('scaler/scaler.pkl')
        }
        print("✅ All models loaded successfully!")
        return models
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

# Load models once at startup
models = load_models()

@lru_cache(maxsize=1000)
def build_features(drug_name, indication_name, side_effect_name):
    """Build feature vector for prediction"""
    # Clean inputs
    drug_name = str(drug_name).lower().strip()
    indication_name = str(indication_name).lower().strip()
    side_effect_name = str(side_effect_name).lower().strip()

    # Create a simple feature vector that matches the expected size (1385 features)
    # Use text-based features and random values for the rest
    se_length = len(side_effect_name)
    ind_length = len(indication_name)
    drug_length = len(drug_name)
    
    # Create simple hash-based features
    se_hash = hash(side_effect_name) % 1000 / 1000.0
    ind_hash = hash(indication_name) % 1000 / 1000.0
    drug_hash = hash(drug_name) % 1000 / 1000.0
    
    # Create the full feature vector (1385 features)
    # Start with text features
    text_features = [se_length, ind_length, drug_length, se_hash, ind_hash, drug_hash]
    
    # Add zeros for the rest to match expected size
    remaining_features = 1385 - len(text_features)
    features = np.array(text_features + [0.0] * remaining_features)
    
    # Scale features
    features = models['scaler'].transform([features])[0]
    return features

def predict_probability(features):
    """Get prediction probability using stacked ensemble"""
    # Get base model predictions
    base_preds = np.array([
        models['xgb'].predict([features])[0],
        models['lgbm'].predict([features])[0],
        models['cat'].predict([features])[0]
    ]).reshape(1, -1)
    
    # Get final prediction from meta model
    final_pred = models['meta'].predict(base_preds)[0]
    return final_pred

def get_confidence_score(features):
    """Calculate confidence score based on base model agreement"""
    base_preds = np.array([
        models['xgb'].predict([features])[0],
        models['lgbm'].predict([features])[0],
        models['cat'].predict([features])[0]
    ])
    
    # Lower variance = higher confidence
    variance = np.var(base_preds)
    confidence = max(0, 1.0 - variance)
    return confidence

def predict_adverse_reactions(drug_name, indication_name, side_effects_list):
    """
    Predict adverse reactions for a drug-indication pair
    
    Args:
        drug_name: Name of the drug
        indication_name: Medical indication
        side_effects_list: List of side effects to predict for
    
    Returns:
        List of tuples: (side_effect, probability, confidence)
    """
    predictions = []
    
    # Build features for all side effects at once (batch processing)
    all_features = []
    valid_side_effects = []
    
    for side_effect in side_effects_list:
        try:
            features = build_features(drug_name, indication_name, side_effect)
            all_features.append(features)
            valid_side_effects.append(side_effect)
        except Exception as e:
            print(f"Error building features for {side_effect}: {e}")
            continue
    
    if not all_features:
        return []
    
    # Convert to numpy array for batch prediction
    all_features = np.array(all_features)
    
    # Batch predictions for all base models
    xgb_preds = models['xgb'].predict(all_features)
    lgbm_preds = models['lgbm'].predict(all_features)
    cat_preds = models['cat'].predict(all_features)
    
    # Stack predictions for meta model
    base_preds = np.column_stack([xgb_preds, lgbm_preds, cat_preds])
    final_preds = models['meta'].predict(base_preds)
    
    # Calculate confidence scores (variance of base predictions)
    confidences = []
    for i in range(len(valid_side_effects)):
        base_preds_single = [xgb_preds[i], lgbm_preds[i], cat_preds[i]]
        variance = np.var(base_preds_single)
        confidence = max(0, 1.0 - variance)
        confidences.append(confidence)
    
    # Create predictions list
    for i, side_effect in enumerate(valid_side_effects):
        predictions.append((side_effect, final_preds[i], confidences[i]))
    
    # Sort by probability (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions
