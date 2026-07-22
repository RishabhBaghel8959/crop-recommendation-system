import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json

class CropRecommendationModel:
    def __init__(self):
        self.crop_model = RandomForestClassifier(random_state=42)
        self.variety_model = RandomForestClassifier(random_state=42)
        self.crop_label_encoder = LabelEncoder()
        self.variety_label_encoder = LabelEncoder()
        self.feature_names = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
        self.crop_variety_mapping = {}
        self.crop_characteristics = {
            'rice': {
                'optimal_conditions': 'High humidity (80-85%), moderate to high rainfall (200-250mm), warm temperature (20-25°C)',
                'benefits': 'High yield potential, staple food crop, good market demand',
                'soil_preference': 'Well-drained loamy soil with pH 6.0-7.0'
            },
            'wheat': {
                'optimal_conditions': 'Moderate humidity (60-70%), low to moderate rainfall (150-200mm), cool temperature (15-20°C)',
                'benefits': 'Drought tolerant, high protein content, versatile crop',
                'soil_preference': 'Well-drained soil with pH 6.0-7.5'
            },
            'cotton': {
                'optimal_conditions': 'Moderate humidity (65-75%), moderate rainfall (180-220mm), warm temperature (25-30°C)',
                'benefits': 'High economic value, industrial use, long growing season',
                'soil_preference': 'Deep, well-drained black cotton soil with pH 6.5-8.0'
            },
            'maize': {
                'optimal_conditions': 'Moderate humidity (60-70%), moderate rainfall (160-200mm), warm temperature (22-28°C)',
                'benefits': 'Fast growing, multiple uses (food/feed/industrial), high yield',
                'soil_preference': 'Well-drained fertile soil with pH 6.0-7.5'
            }
        }
        
        self.variety_characteristics = {
            'ir64': 'High-yielding variety with good disease resistance and medium maturity',
            'swarna': 'Premium quality variety with excellent cooking quality and aroma',
            'savitri': 'Early maturing variety suitable for water-scarce regions',
            'pusa': 'High-yielding variety with excellent grain quality and market value'
        }

    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the crop recommendation dataset"""
        try:
            
            try:
                df = pd.read_csv("D:\\Ankita Project\\Crop_recommendation.csv"
)
            except UnicodeDecodeError:
                df = pd.read_csv("D:\\Ankita Project\\Crop_recommendation.csv", encoding='latin-1')
            
            print(f"Dataset loaded successfully with shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Rename columns to match expected format
            column_mapping = {
                'nitrogen': 'n', 'phosphorus': 'p', 'potassium': 'k',
                'temp': 'temperature', 'humid': 'humidity'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure all required columns exist
            required_cols = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall', 'label', 'variety']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Build crop-variety mapping
            for _, row in df.iterrows():
                crop = row['label']
                variety = row['variety']
                if crop not in self.crop_variety_mapping:
                    self.crop_variety_mapping[crop] = []
                if variety not in self.crop_variety_mapping[crop]:
                    self.crop_variety_mapping[crop].append(variety)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Generate synthetic data if file not found
            return self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """Generate synthetic crop data based on the dataset structure"""
        np.random.seed(42)
        n_samples = 1000
        
        # Define crop-variety combinations based on your dataset
        crop_varieties = {
            'rice': ['ir64', 'swarna', 'savitri'],
            'wheat': ['pusa', 'hd2967', 'dbw88'],
            'cotton': ['bt', 'desi', 'hybrid'],
            'maize': ['single_cross', 'composite', 'hybrid']
        }
        
        data = []
        for crop, varieties in crop_varieties.items():
            for variety in varieties:
                n_crop_samples = n_samples // (len(crop_varieties) * len(varieties))
                
                if crop == 'rice':
                    # Rice prefers high humidity, moderate to high rainfall
                    n_vals = np.random.normal(85, 10, n_crop_samples)
                    p_vals = np.random.normal(50, 8, n_crop_samples)
                    k_vals = np.random.normal(40, 5, n_crop_samples)
                    temp_vals = np.random.normal(22, 3, n_crop_samples)
                    humidity_vals = np.random.normal(82, 3, n_crop_samples)
                    ph_vals = np.random.normal(6.5, 0.5, n_crop_samples)
                    rainfall_vals = np.random.normal(225, 25, n_crop_samples)
                    
                elif crop == 'wheat':
                    # Wheat prefers cooler, drier conditions
                    n_vals = np.random.normal(75, 8, n_crop_samples)
                    p_vals = np.random.normal(45, 6, n_crop_samples)
                    k_vals = np.random.normal(35, 4, n_crop_samples)
                    temp_vals = np.random.normal(18, 2, n_crop_samples)
                    humidity_vals = np.random.normal(65, 5, n_crop_samples)
                    ph_vals = np.random.normal(7.0, 0.4, n_crop_samples)
                    rainfall_vals = np.random.normal(175, 20, n_crop_samples)
                    
                elif crop == 'cotton':
                    # Cotton prefers warm conditions with moderate water
                    n_vals = np.random.normal(80, 9, n_crop_samples)
                    p_vals = np.random.normal(55, 7, n_crop_samples)
                    k_vals = np.random.normal(45, 6, n_crop_samples)
                    temp_vals = np.random.normal(27, 3, n_crop_samples)
                    humidity_vals = np.random.normal(70, 4, n_crop_samples)
                    ph_vals = np.random.normal(7.2, 0.6, n_crop_samples)
                    rainfall_vals = np.random.normal(200, 22, n_crop_samples)
                    
                else:  # maize
                    # Maize prefers warm conditions with good nutrition
                    n_vals = np.random.normal(88, 10, n_crop_samples)
                    p_vals = np.random.normal(52, 8, n_crop_samples)
                    k_vals = np.random.normal(42, 5, n_crop_samples)
                    temp_vals = np.random.normal(25, 3, n_crop_samples)
                    humidity_vals = np.random.normal(68, 4, n_crop_samples)
                    ph_vals = np.random.normal(6.8, 0.5, n_crop_samples)
                    rainfall_vals = np.random.normal(185, 18, n_crop_samples)
                
                # Ensure positive values
                n_vals = np.clip(n_vals, 10, 150)
                p_vals = np.clip(p_vals, 5, 100)
                k_vals = np.clip(k_vals, 5, 80)
                temp_vals = np.clip(temp_vals, 10, 40)
                humidity_vals = np.clip(humidity_vals, 20, 95)
                ph_vals = np.clip(ph_vals, 4.0, 9.0)
                rainfall_vals = np.clip(rainfall_vals, 50, 400)
                
                for i in range(n_crop_samples):
                    data.append({
                        'n': round(n_vals[i], 2),
                        'p': round(p_vals[i], 2),
                        'k': round(k_vals[i], 2),
                        'temperature': round(temp_vals[i], 2),
                        'humidity': round(humidity_vals[i], 2),
                        'ph': round(ph_vals[i], 2),
                        'rainfall': round(rainfall_vals[i], 2),
                        'label': crop,
                        'variety': variety
                    })
        
        df = pd.DataFrame(data)
        
        # Update crop-variety mapping
        for crop, varieties in crop_varieties.items():
            self.crop_variety_mapping[crop] = varieties
            
        print(f"Generated synthetic dataset with shape: {df.shape}")
        return df

    def train_models(self, df):
        """Train both crop and variety prediction models"""
        # Prepare features
        X = df[self.feature_names].values
        y_crop = df['label'].values
        y_variety = df['variety'].values
        
        # Encode labels
        y_crop_encoded = self.crop_label_encoder.fit_transform(y_crop)
        y_variety_encoded = self.variety_label_encoder.fit_transform(y_variety)
        
        # Split data
        X_train, X_test, y_crop_train, y_crop_test, y_variety_train, y_variety_test = train_test_split(
            X, y_crop_encoded, y_variety_encoded, test_size=0.2, random_state=42, stratify=y_crop_encoded
        )
        
        # Hyperparameter tuning for crop model
        crop_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print("Tuning crop prediction model...")
        crop_grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            crop_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        crop_grid_search.fit(X_train, y_crop_train)
        self.crop_model = crop_grid_search.best_estimator_
        
        # Hyperparameter tuning for variety model
        print("Tuning variety prediction model...")
        variety_grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            crop_param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        variety_grid_search.fit(X_train, y_variety_train)
        self.variety_model = variety_grid_search.best_estimator_
        
        # Evaluate models
        crop_pred = self.crop_model.predict(X_test)
        variety_pred = self.variety_model.predict(X_test)
        
        crop_accuracy = accuracy_score(y_crop_test, crop_pred)
        variety_accuracy = accuracy_score(y_variety_test, variety_pred)
        
        print(f"\nCrop Model Accuracy: {crop_accuracy:.4f}")
        print(f"Variety Model Accuracy: {variety_accuracy:.4f}")
        
        print(f"\nBest Crop Model Parameters: {crop_grid_search.best_params_}")
        print(f"Best Variety Model Parameters: {variety_grid_search.best_params_}")
        
        # Cross-validation scores
        crop_cv_scores = cross_val_score(self.crop_model, X, y_crop_encoded, cv=5)
        variety_cv_scores = cross_val_score(self.variety_model, X, y_variety_encoded, cv=5)
        
        print(f"\nCrop Model CV Accuracy: {crop_cv_scores.mean():.4f} (+/- {crop_cv_scores.std() * 2:.4f})")
        print(f"Variety Model CV Accuracy: {variety_cv_scores.mean():.4f} (+/- {variety_cv_scores.std() * 2:.4f})")
        
        return {
            'crop_accuracy': crop_accuracy,
            'variety_accuracy': variety_accuracy,
            'crop_cv_mean': crop_cv_scores.mean(),
            'variety_cv_mean': variety_cv_scores.mean()
        }

    def predict_crop_and_variety(self, n, p, k, temperature, humidity, ph, rainfall):
        """Predict crop and variety with detailed explanations"""
        # Prepare input features
        features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        
        # Predict crop
        crop_pred_encoded = self.crop_model.predict(features)[0]
        crop_pred = self.crop_label_encoder.inverse_transform([crop_pred_encoded])[0]
        crop_probabilities = self.crop_model.predict_proba(features)[0]
        crop_confidence = max(crop_probabilities)
        
        # Predict variety
        variety_pred_encoded = self.variety_model.predict(features)[0]
        variety_pred = self.variety_label_encoder.inverse_transform([variety_pred_encoded])[0]
        variety_probabilities = self.variety_model.predict_proba(features)[0]
        variety_confidence = max(variety_probabilities)
        
        # Get feature importance
        crop_feature_importance = dict(zip(self.feature_names, self.crop_model.feature_importances_))
        variety_feature_importance = dict(zip(self.feature_names, self.variety_model.feature_importances_))
        
        # Generate detailed explanation
        explanation = self.generate_explanation(
            crop_pred, variety_pred, n, p, k, temperature, humidity, ph, rainfall,
            crop_feature_importance, variety_feature_importance, crop_confidence, variety_confidence
        )
        
        return {
            'crop': crop_pred,
            'variety': variety_pred,
            'crop_confidence': round(crop_confidence * 100, 2),
            'variety_confidence': round(variety_confidence * 100, 2),
            'explanation': explanation,
            'feature_importance': {
                'crop': {k: round(v, 4) for k, v in crop_feature_importance.items()},
                'variety': {k: round(v, 4) for k, v in variety_feature_importance.items()}
            }
        }

    def generate_explanation(self, crop, variety, n, p, k, temp, humidity, ph, rainfall, 
                           crop_importance, variety_importance, crop_conf, variety_conf):
        """Generate detailed explanation for crop and variety recommendation"""
        
        explanation = {
            'recommendation_summary': f"Based on your soil and environmental conditions, we recommend cultivating {crop.upper()} with {variety.upper()} variety.",
            'confidence_scores': {
                'crop_confidence': f"{crop_conf*100:.1f}%",
                'variety_confidence': f"{variety_conf*100:.1f}%"
            },
            'why_this_crop': [],
            'why_this_variety': [],
            'condition_analysis': [],
            'expected_benefits': [],
            'cultivation_tips': []
        }
        
        # Analyze soil nutrients
        if n > 80:
            explanation['condition_analysis'].append(f"High nitrogen content ({n}) is excellent for {crop} growth and leaf development.")
        elif n < 50:
            explanation['condition_analysis'].append(f"Low nitrogen content ({n}) may require additional fertilization for optimal {crop} yield.")
        else:
            explanation['condition_analysis'].append(f"Moderate nitrogen content ({n}) is suitable for {crop} cultivation.")
            
        if p > 45:
            explanation['condition_analysis'].append(f"Good phosphorus levels ({p}) will support strong root development and flowering.")
        elif p < 30:
            explanation['condition_analysis'].append(f"Low phosphorus ({p}) may need supplementation for better root growth.")
            
        if k > 40:
            explanation['condition_analysis'].append(f"Adequate potassium ({k}) will enhance disease resistance and fruit quality.")
            
        # Analyze environmental conditions
        if crop in self.crop_characteristics:
            crop_info = self.crop_characteristics[crop]
            explanation['why_this_crop'].append(f"Environmental conditions match {crop} requirements: {crop_info['optimal_conditions']}")
            explanation['expected_benefits'].append(crop_info['benefits'])
            explanation['cultivation_tips'].append(f"Soil preference: {crop_info['soil_preference']}")
            
        if variety in self.variety_characteristics:
            explanation['why_this_variety'].append(self.variety_characteristics[variety])
            
        # Temperature analysis
        if crop == 'rice' and 20 <= temp <= 25:
            explanation['condition_analysis'].append(f"Temperature ({temp}°C) is optimal for rice growth and grain filling.")
        elif crop == 'wheat' and 15 <= temp <= 20:
            explanation['condition_analysis'].append(f"Cool temperature ({temp}°C) is perfect for wheat development.")
        elif crop == 'cotton' and 25 <= temp <= 30:
            explanation['condition_analysis'].append(f"Warm temperature ({temp}°C) favors cotton fiber development.")
        elif crop == 'maize' and 22 <= temp <= 28:
            explanation['condition_analysis'].append(f"Temperature ({temp}°C) is ideal for maize growth and yield.")
            
        # Humidity analysis
        if humidity > 80 and crop == 'rice':
            explanation['condition_analysis'].append(f"High humidity ({humidity}%) is perfect for rice cultivation.")
        elif 60 <= humidity <= 75 and crop in ['wheat', 'maize']:
            explanation['condition_analysis'].append(f"Moderate humidity ({humidity}%) suits {crop} well.")
            
        # pH analysis
        if 6.0 <= ph <= 7.5:
            explanation['condition_analysis'].append(f"Soil pH ({ph}) is in the optimal range for most crops.")
        elif ph < 6.0:
            explanation['condition_analysis'].append(f"Slightly acidic soil (pH {ph}) may need lime application.")
        elif ph > 7.5:
            explanation['condition_analysis'].append(f"Alkaline soil (pH {ph}) may need organic matter addition.")
            
        # Rainfall analysis
        if rainfall > 200 and crop == 'rice':
            explanation['condition_analysis'].append(f"High rainfall ({rainfall}mm) is excellent for rice cultivation.")
        elif 150 <= rainfall <= 200 and crop in ['wheat', 'maize']:
            explanation['condition_analysis'].append(f"Moderate rainfall ({rainfall}mm) is suitable for {crop}.")
            
        # Feature importance insights
        most_important_feature = max(crop_importance, key=crop_importance.get)
        explanation['why_this_crop'].append(f"The model primarily considers {most_important_feature} (importance: {crop_importance[most_important_feature]:.3f}) for crop selection.")
        
        variety_important_feature = max(variety_importance, key=variety_importance.get)
        explanation['why_this_variety'].append(f"For variety selection, {variety_important_feature} (importance: {variety_importance[variety_important_feature]:.3f}) is the key factor.")
        
        # Additional cultivation tips
        explanation['cultivation_tips'].extend([
            "Monitor soil moisture regularly and maintain optimal irrigation.",
            "Consider crop rotation to maintain soil health.",
            "Use appropriate fertilization based on soil test results.",
            "Implement pest and disease management practices."
        ])
        
        return explanation

    def save_models(self, model_dir='models'):
        """Save trained models and encoders"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.crop_model, f'{model_dir}/crop_model.pkl')
        joblib.dump(self.variety_model, f'{model_dir}/variety_model.pkl')
        joblib.dump(self.crop_label_encoder, f'{model_dir}/crop_label_encoder.pkl')
        joblib.dump(self.variety_label_encoder, f'{model_dir}/variety_label_encoder.pkl')
        
        # Save crop-variety mapping
        with open(f'{model_dir}/crop_variety_mapping.json', 'w') as f:
            json.dump(self.crop_variety_mapping, f)
            
        print(f"Models saved to {model_dir}/ directory")

    def load_models(self, model_dir='models'):
        """Load trained models and encoders"""
        self.crop_model = joblib.load(f'{model_dir}/crop_model.pkl')
        self.variety_model = joblib.load(f'{model_dir}/variety_model.pkl')
        self.crop_label_encoder = joblib.load(f'{model_dir}/crop_label_encoder.pkl')
        self.variety_label_encoder = joblib.load(f'{model_dir}/variety_label_encoder.pkl')
        
        # Load crop-variety mapping
        with open(f'{model_dir}/crop_variety_mapping.json', 'r') as f:
            self.crop_variety_mapping = json.load(f)
            
        print("Models loaded successfully")




if __name__ == "__main__":
    # Create an instance of your model class
    model = CropRecommendationModel()

    # Load and preprocess your dataset
    df = model.load_and_preprocess_data("D:\\Ankita Project\\Crop_recommendation.csv")

    # OPTIONAL: Train your models (if not already done elsewhere)
    model.train_models(df)

    # OPTIONAL: Make a test prediction (fill with sample realistic values)
    result = model.predict_crop_and_variety(
        n=90, p=45, k=38, temperature=25, humidity=80, ph=6.8, rainfall=220
    )
    print(result)
