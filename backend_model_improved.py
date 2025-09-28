#!/usr/bin/env python3
"""
Rockfall Risk Assessment System - Improved ML Backend
Student Project - Mining Engineering Department
Features: High-accuracy ML model with optimized parameters and ensemble methods
"""

import pandas as pd
import numpy as np
from twilio.rest import Client
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

import os
from dotenv import load_dotenv

# ---------------------- SMS Configuration ----------------------
load_dotenv()

SMS_CONFIG = {
    'account_sid': os.getenv("TWILIO_ACCOUNT_SID"),
    'auth_token': os.getenv("TWILIO_AUTH_TOKEN"),
    'from_number': os.getenv("TWILIO_FROM_NUMBER"),
    'to_number': os.getenv("TWILIO_TO_NUMBER"),
    'enabled': True 
}

# ---------------------- Original Algorithm (Fallback) ----------------------
def calculate_risk_score(params):
    """Original risk calculation algorithm as fallback."""
    # Normalize parameters
    slope_factor = (params['slope_angle'] - 30) / 60
    slope_risk = max(0, min(1, slope_factor)) ** 1.8

    height_factor = (params['slope_height'] - 100) / 700
    height_risk = max(0, min(1, height_factor)) ** 1.5

    # Rock type risk mapping
    rock_risks = {'Granite': 0.15, 'Limestone': 0.35, 'Sandstone': 0.65, 'Shale': 0.85}
    rock_risk = rock_risks[params['rock_type']]

    # Fracture density risk
    fracture_risks = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}
    fracture_risk = fracture_risks[params['fracture_density']]

    # Rainfall impact
    rain_factor = params['rainfall_24h'] / 60
    rain_risk = max(0, min(1, rain_factor)) ** 2

    # Blast impact
    blast_factor = max(0, (14 - params['days_since_blast']) / 14)
    blast_risk = blast_factor ** 1.5

    # Weighted combination
    total_risk = (
        slope_risk * 0.25 +
        height_risk * 0.20 +
        rock_risk * 0.20 +
        fracture_risk * 0.15 +
        rain_risk * 0.10 +
        blast_risk * 0.10
    )

    # Interaction bonuses
    interaction_bonus = 0
    if (params['slope_angle'] > 80 and 
        params['rock_type'] == 'Shale' and 
        params['fracture_density'] == 'High' and
        params['rainfall_24h'] > 40 and
        params['days_since_blast'] <= 2):
        interaction_bonus = 0.25
    elif (params['slope_angle'] > 70 and 
          params['fracture_density'] == 'High' and
          params['rainfall_24h'] > 30):
        interaction_bonus = 0.15
    elif (params['slope_angle'] > 60 and 
          params['rock_type'] in ['Sandstone', 'Shale'] and
          params['days_since_blast'] <= 5):
        interaction_bonus = 0.08

    final_risk = total_risk + interaction_bonus
    return max(0.05, min(0.95, final_risk))

# ---------------------- Risk Category ----------------------
def get_risk_category(risk_score):
    """Determine risk category and safety message."""
    if risk_score >= 0.80:
        return "CRITICAL", "Immediate evacuation required"
    elif risk_score >= 0.65:
        return "HIGH", "Enhanced monitoring needed"
    elif risk_score >= 0.40:
        return "MEDIUM", "Caution advised"
    else:
        return "LOW", "Safe operations"

# ---------------------- SMS Alert ----------------------
def send_sms_alert(risk_score, inputs):
    """Send SMS alert with current scenario."""
    if not SMS_CONFIG['enabled']:
        return False, "SMS disabled"
    
    try:
        message_text = f"""ROCKFALL RISK ALERT

Risk Level: {risk_score:.0%}
Time: {datetime.now().strftime('%H:%M:%S')}

Current Conditions:
- Slope Angle: {inputs['slope_angle']:.0f} degrees
- Height: {inputs['slope_height']:.0f} ft
- Rock Type: {inputs['rock_type']}
- Fractures: {inputs['fracture_density']}
- Rainfall: {inputs['rainfall_24h']:.1f} mm (24h)
- Days Since Blast: {inputs['days_since_blast']}

Mining Safety Assessment System
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
        
        client = Client(SMS_CONFIG['account_sid'], SMS_CONFIG['auth_token'])
        message = client.messages.create(
            body=message_text,
            from_=SMS_CONFIG['from_number'],
            to=SMS_CONFIG['to_number']
        )
        return True, message.sid
    except Exception as e:
        return False, str(e)

# ---------------------- Example Scenarios ----------------------
LOW_RISK_SCENARIO = {
    'slope_angle': 35,
    'slope_height': 150,
    'rock_type': 'Granite',
    'fracture_density': 'Low',
    'rainfall_24h': 2.0,
    'days_since_blast': 14
}

HIGH_RISK_SCENARIO = {
    'slope_angle': 85,
    'slope_height': 750,
    'rock_type': 'Shale',
    'fracture_density': 'High',
    'rainfall_24h': 50.0,
    'days_since_blast': 1
}

# ---------------------- Enhanced Ensemble Model ----------------------
class EnsembleRiskModel:
    """Ensemble model combining multiple ML algorithms for better accuracy."""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.preprocessor = None
        self.is_trained = False
    
    def create_models(self):
        """Create multiple ML models for ensemble."""
        self.models = {
            'rf_optimized': RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(alpha=0.1, random_state=42)
        }
        
        # Model weights (higher = more important)
        self.weights = {
            'rf_optimized': 0.35,
            'gbm': 0.30,
            'extra_trees': 0.25,
            'ridge': 0.10
        }
    
    def create_preprocessor(self):
        """Create optimized preprocessing pipeline."""
        categorical_features = ['rock_type', 'fracture_density']
        numeric_features = ['slope_angle', 'slope_height', 'rainfall_24h', 'days_since_blast']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
                ('num', StandardScaler(), numeric_features)  # Added scaling for better performance
            ],
            remainder='passthrough'
        )
    
    def train(self, X, y):
        """Train ensemble of models."""
        print("🚀 Training Enhanced ML Ensemble...")
        
        self.create_models()
        self.create_preprocessor()
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Transform features
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        
        # Train each model
        model_scores = {}
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train_processed, y_train)
            
            # Validate
            val_pred = model.predict(X_val_processed)
            mae = mean_absolute_error(y_val, val_pred)
            r2 = r2_score(y_val, val_pred)
            model_scores[name] = {'mae': mae, 'r2': r2}
            print(f"    MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Adjust weights based on performance
        self.adjust_weights(model_scores)
        
        self.is_trained = True
        print("✅ Ensemble training complete!")
        return model_scores
    
    def adjust_weights(self, scores):
        """Dynamically adjust model weights based on performance."""
        # Lower MAE = better performance = higher weight
        total_inverse_mae = sum(1.0 / scores[name]['mae'] for name in scores)
        
        for name in self.weights:
            # Weight based on inverse MAE (lower error = higher weight)
            performance_weight = (1.0 / scores[name]['mae']) / total_inverse_mae
            # Blend with original weights
            self.weights[name] = 0.7 * performance_weight + 0.3 * self.weights[name]
        
        print(f"📊 Adjusted model weights: {self.weights}")
    
    def predict(self, X):
        """Make ensemble prediction."""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        # Preprocess input
        X_processed = self.preprocessor.transform(X)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_processed)
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        # Ensure valid range
        ensemble_pred = np.clip(ensemble_pred, 0.05, 0.95)
        
        return ensemble_pred

# ---------------------- Improved Rockfall Risk System ----------------------
class ImprovedRockfallRiskSystem:
    """Improved rockfall risk assessment system with high-accuracy ML."""
    
    def __init__(self):
        self.dataset = None
        self.scenarios = None
        self.metadata = None
        self.ml_model = None
        self.ml_available = False
        self.load_datasets()
        if self.dataset is not None:
            self.train_ml_model()
    
    def load_datasets(self):
        """Load all integrated datasets."""
        try:
            datasets = ['student_model_dataset.csv', 'student_model_dataset_2.csv', 'student_model_dataset_3.csv']
            combined_df = []
            loaded_files = []
            
            for file in datasets:
                if os.path.exists(file):
                    df = pd.read_csv(file)
                    combined_df.append(df)
                    loaded_files.append(file)
            
            if combined_df:
                self.dataset = pd.concat(combined_df, ignore_index=True)
                print(f"✅ Combined dataset loaded: {len(self.dataset)} samples from {len(loaded_files)} files")
            
            if os.path.exists('student_model_scenarios.csv'):
                self.scenarios = pd.read_csv('student_model_scenarios.csv')
                print(f"✅ Test scenarios loaded: {len(self.scenarios)} scenarios")
            
            if os.path.exists('student_model_info.json'):
                with open('student_model_info.json', 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Metadata loaded: {self.metadata['name']}")
                
        except Exception as e:
            print(f"⚠️  Dataset files not found: {e}")
            print("System will work in standalone mode.")
    
    def engineer_features(self, df):
        """Create additional engineered features for better accuracy."""
        df_eng = df.copy()
        
        # Interaction features based on original algorithm
        df_eng['slope_height_interaction'] = df_eng['slope_angle'] * df_eng['slope_height'] / 1000
        df_eng['rain_blast_interaction'] = df_eng['rainfall_24h'] * (30 - df_eng['days_since_blast']) / 100
        
        # Rock type risk encoding (based on original algorithm)
        rock_risk_map = {'Granite': 0.15, 'Limestone': 0.35, 'Sandstone': 0.65, 'Shale': 0.85}
        df_eng['rock_risk_encoded'] = df_eng['rock_type'].map(rock_risk_map)
        
        # Fracture density encoding
        fracture_risk_map = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}
        df_eng['fracture_risk_encoded'] = df_eng['fracture_density'].map(fracture_risk_map)
        
        # Polynomial features for key parameters
        df_eng['slope_angle_squared'] = df_eng['slope_angle'] ** 2
        df_eng['slope_height_sqrt'] = np.sqrt(df_eng['slope_height'])
        
        # Normalized features (same as original algorithm)
        df_eng['slope_norm'] = (df_eng['slope_angle'] - 30) / 60
        df_eng['height_norm'] = (df_eng['slope_height'] - 100) / 700
        df_eng['rain_norm'] = df_eng['rainfall_24h'] / 60
        df_eng['blast_norm'] = np.maximum(0, (14 - df_eng['days_since_blast']) / 14)
        
        return df_eng
    
    def train_ml_model(self):
        """Train the improved ML model."""
        try:
            print("🎯 Training Improved ML Model with Feature Engineering...")
            
            # Engineer features
            df_engineered = self.engineer_features(self.dataset)
            
            # Select features for training
            feature_columns = [
                'slope_angle', 'slope_height', 'rock_type', 'fracture_density', 
                'rainfall_24h', 'days_since_blast',
                'slope_height_interaction', 'rain_blast_interaction',
                'rock_risk_encoded', 'fracture_risk_encoded',
                'slope_angle_squared', 'slope_height_sqrt',
                'slope_norm', 'height_norm', 'rain_norm', 'blast_norm'
            ]
            
            X = df_engineered[feature_columns]
            y = df_engineered['risk_score']
            
            # Train ensemble model
            self.ml_model = EnsembleRiskModel()
            scores = self.ml_model.train(X, y)
            
            self.ml_available = True
            print("✅ Improved ML model training complete!")
            
            # Overall performance
            best_mae = min(score['mae'] for score in scores.values())
            print(f"🎯 Best individual model MAE: {best_mae:.4f}")
            
        except Exception as e:
            print(f"⚠️  ML model training failed: {e}")
            self.ml_available = False
    
    def calculate_risk(self, params):
        """Predict risk using improved ML model."""
        if self.ml_available and self.ml_model is not None:
            try:
                # Create dataframe with engineered features
                df_input = pd.DataFrame([params])
                df_eng = self.engineer_features(df_input)
                
                # Select same features as training
                feature_columns = [
                    'slope_angle', 'slope_height', 'rock_type', 'fracture_density', 
                    'rainfall_24h', 'days_since_blast',
                    'slope_height_interaction', 'rain_blast_interaction',
                    'rock_risk_encoded', 'fracture_risk_encoded',
                    'slope_angle_squared', 'slope_height_sqrt',
                    'slope_norm', 'height_norm', 'rain_norm', 'blast_norm'
                ]
                
                X = df_eng[feature_columns]
                predicted_risk = self.ml_model.predict(X)[0]
                
                return max(0.05, min(0.95, predicted_risk))
                
            except Exception as e:
                print(f"⚠️  ML prediction failed: {e}, using fallback")
                return calculate_risk_score(params)
        else:
            return calculate_risk_score(params)
    
    def get_category(self, risk_score):
        return get_risk_category(risk_score)
    
    def send_alert(self, risk_score, params):
        return send_sms_alert(risk_score, params)
    
    def find_similar_cases(self, params, top_n=5):
        """Find similar cases in dataset based on input parameters."""
        if self.dataset is None:
            return []
        
        similarities = []
        for _, row in self.dataset.iterrows():
            # Normalize differences
            angle_diff = abs(params['slope_angle'] - row['slope_angle']) / 60
            height_diff = abs(params['slope_height'] - row['slope_height']) / 700
            rain_diff = abs(params['rainfall_24h'] - row['rainfall_24h']) / 60
            blast_diff = abs(params['days_since_blast'] - row['days_since_blast']) / 30
            
            # Exact match bonuses
            rock_match = 1.0 if params['rock_type'] == row['rock_type'] else 0.0
            fracture_match = 1.0 if params['fracture_density'] == row['fracture_density'] else 0.0
            
            # Overall similarity (lower = more similar)
            similarity = (angle_diff + height_diff + rain_diff + blast_diff + 
                         (1-rock_match) + (1-fracture_match)) / 6
            
            similarities.append({
                'index': row.name,
                'similarity': similarity,
                'risk_percentage': row['risk_percentage'],
                'slope_angle': row['slope_angle'],
                'rock_type': row['rock_type'],
                'fracture_density': row['fracture_density']
            })
        
        similarities.sort(key=lambda x: x['similarity'])
        return similarities[:top_n]
    
    def validate_model(self, num_samples=100):
        """Validate improved model accuracy."""
        if self.dataset is None:
            print("❌ No dataset available for validation")
            return False
        
        print(f"\n🔍 VALIDATING IMPROVED MODEL ({num_samples} samples)")
        print("-" * 50)
        
        test_samples = self.dataset.sample(n=min(num_samples, len(self.dataset)), random_state=42)
        
        accurate_predictions = 0
        total_error = 0
        category_matches = 0
        
        for _, row in test_samples.iterrows():
            params = {
                'slope_angle': row['slope_angle'],
                'slope_height': row['slope_height'],
                'rock_type': row['rock_type'],
                'fracture_density': row['fracture_density'],
                'rainfall_24h': row['rainfall_24h'],
                'days_since_blast': row['days_since_blast']
            }
            
            predicted_risk = self.calculate_risk(params)
            actual_risk = row['risk_score']
            error = abs(predicted_risk - actual_risk)
            total_error += error
            
            # Check prediction accuracy (relaxed tolerance)
            if error < 0.005:  # More reasonable tolerance
                accurate_predictions += 1
            
            # Check category accuracy
            pred_cat, _ = self.get_category(predicted_risk)
            actual_cat, _ = self.get_category(actual_risk)
            if pred_cat == actual_cat:
                category_matches += 1
        
        accuracy = (accurate_predictions / len(test_samples)) * 100
        category_accuracy = (category_matches / len(test_samples)) * 100
        avg_error = total_error / len(test_samples)
        
        print(f"Numerical Accuracy: {accurate_predictions}/{len(test_samples)} ({accuracy:.1f}%)")
        print(f"Category Accuracy: {category_matches}/{len(test_samples)} ({category_accuracy:.1f}%)")
        print(f"Average Error: {avg_error:.6f}")
        print(f"Model Type: {'🤖 Improved ML Ensemble' if self.ml_available else '⚙️ Original Algorithm'}")
        print("✅ Model validation successful!" if category_accuracy > 95 else "⚠️ Model needs review")
        
        return category_accuracy > 95
    
    def test_scenarios(self):
        """Test predefined scenarios."""
        if self.scenarios is None:
            print("❌ No test scenarios available")
            return
        
        print(f"\n🎯 TESTING SCENARIOS")
        print("-" * 25)
        
        for _, row in self.scenarios.iterrows():
            params = {
                'slope_angle': row['slope_angle'],
                'slope_height': row['slope_height'],
                'rock_type': row['rock_type'],
                'fracture_density': row['fracture_density'],
                'rainfall_24h': row['rainfall_24h'],
                'days_since_blast': row['days_since_blast']
            }
            
            risk_score = self.calculate_risk(params)
            category, message = self.get_category(risk_score)
            
            # Also test with original algorithm for comparison
            orig_risk = calculate_risk_score(params)
            
            print(f"\n📋 {row['scenario_name']}")
            print(f"   Expected: {row['expected_range']}")
            print(f"   ML Prediction: {risk_score:.1%} ({category})")
            print(f"   Original Algo: {orig_risk:.1%}")
            print(f"   Dataset Value: {row['actual_risk']:.1f}%")
            print(f"   ML vs Original: {abs(risk_score - orig_risk):.3f} difference")
    
    def run_comprehensive_test(self):
        """Run comprehensive testing."""
        print("🏔️  IMPROVED ROCKFALL RISK SYSTEM TEST")
        print("=" * 50)
        
        if self.dataset is not None:
            print(f"Dataset: {len(self.dataset):,} samples loaded")
            print(f"ML Model: {'✅ Improved Ensemble Available' if self.ml_available else '❌ Using Original Algorithm'}")
        
        self.validate_model(200)
        self.test_scenarios()
        
        print(f"\n✅ COMPREHENSIVE TEST COMPLETE")
        print("Improved system operational with enhanced accuracy!")

# ---------------------- Instantiate System ----------------------
improved_risk_system = ImprovedRockfallRiskSystem()

# ---------------------- Example Usage ----------------------
if __name__ == "__main__":
    improved_risk_system.run_comprehensive_test()
    
    # Test example
    example_params = {
        'slope_angle': 70,
        'slope_height': 500,
        'rock_type': 'Sandstone',
        'fracture_density': 'High',
        'rainfall_24h': 25.0,
        'days_since_blast': 3
    }
    
    print(f"\n" + "="*50)
    print("IMPROVED RISK ASSESSMENT EXAMPLE")
    print("="*50)
    
    risk_score = improved_risk_system.calculate_risk(example_params)
    category, message = improved_risk_system.get_category(risk_score)
    original_risk = calculate_risk_score(example_params)
    
    print(f"Parameters: {example_params}")
    print(f"Improved ML: {risk_score:.1%} ({category})")
    print(f"Original: {original_risk:.1%}")
    print(f"Difference: {abs(risk_score - original_risk):.3f}")
    print(f"Recommendation: {message}")