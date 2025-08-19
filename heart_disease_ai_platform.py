#!/usr/bin/env python3
"""
🏥 Heart Disease AI Platform - Clean GitHub Version
Advanced cardiac risk assessment system with explainable AI
Removed all Palantir-specific content for open source distribution
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    import xgboost as xgb
    import shap
    from lime import lime_tabular
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    print(f"Warning: Some ML libraries not available: {e}")
    print("Install with: pip install scikit-learn xgboost shap lime matplotlib seaborn plotly")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cardio_risk_platform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeartDiseaseDataProcessor:
    """
    Advanced data processor for multi-institutional cardiac data
    Handles data fusion, preprocessing, and feature engineering
    """
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_names = []
        self.processed_data = None
        
        # Define the 14 key features used in heart disease prediction
        self.key_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
        ]
        
        # Feature descriptions for explainability
        self.feature_descriptions = {
            'age': 'Patient age in years',
            'sex': 'Patient sex (1=male, 0=female)',
            'cp': 'Chest pain type (1-4)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1=true, 0=false)',
            'restecg': 'Resting ECG results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1=yes, 0=no)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of peak exercise ST segment (1-3)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia type (3=normal, 6=fixed defect, 7=reversible defect)',
            'num': 'Diagnosis of heart disease (0=no, 1-4=yes)'
        }
    
    def load_multi_institutional_data(self) -> pd.DataFrame:
        """
        Load and combine data from all four international institutions
        """
        logger.info("Loading multi-institutional cardiac data...")
        
        datasets = {}
        
        # Load each dataset
        data_files = {
            'cleveland': 'cleveland.data',
            'hungarian': 'hungarian.data', 
            'switzerland': 'switzerland.data',
            'long_beach_va': 'long-beach-va.data'
        }
        
        for name, filename in data_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    # Load raw data (76 columns)
                    data = pd.read_csv(file_path, header=None, sep='\s+')
                    datasets[name] = data
                    logger.info(f"Loaded {name} dataset: {len(data)} patients")
                except Exception as e:
                    logger.error(f"Error loading {name} dataset: {e}")
        
        # Combine all datasets
        if datasets:
            combined_data = pd.concat(datasets.values(), ignore_index=True)
            logger.info(f"Combined dataset: {len(combined_data)} total patients")
            return combined_data
        else:
            logger.error("No datasets found!")
            return pd.DataFrame()
    
    def extract_key_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the 14 key features used in heart disease prediction
        Based on the heart-disease.names documentation
        """
        logger.info("Extracting key features...")
        
        # Column indices for the 14 key features (0-indexed)
        feature_indices = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]
        
        # Extract features and assign names
        extracted_data = data.iloc[:, feature_indices].copy()
        extracted_data.columns = self.key_features
        
        logger.info(f"Extracted {len(extracted_data.columns)} features from {len(extracted_data)} patients")
        return extracted_data
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Comprehensive data preprocessing pipeline
        """
        logger.info("Starting data preprocessing...")
        
        # Handle missing values
        data_clean = data.copy()
        
        # Replace -9 with NaN (missing value indicator in this dataset)
        data_clean = data_clean.replace(-9, np.nan)
        
        # Impute missing values
        for col in data_clean.columns:
            if data_clean[col].isnull().sum() > 0:
                if col == 'num':  # Target variable
                    data_clean[col] = data_clean[col].fillna(0)
                else:
                    data_clean[col] = data_clean[col].fillna(data_clean[col].median())
        
        # Convert target to binary (0 = no heart disease, 1+ = heart disease)
        data_clean['num'] = (data_clean['num'] > 0).astype(int)
        
        # Separate features and target
        X = data_clean.drop('num', axis=1)
        y = data_clean['num']
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        logger.info(f"Preprocessing complete: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        return {
            'total_patients': len(self.processed_data) if self.processed_data is not None else 0,
            'total_features': len(self.key_features) - 1,  # Exclude target
            'data_sources': ['Cleveland Clinic', 'Hungarian Institute', 'V.A. Medical Center', 'University Hospital Zurich'],
            'feature_names': self.key_features[:-1],  # Exclude target
            'feature_descriptions': self.feature_descriptions
        }
    
    def process_pipeline(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete data processing pipeline"""
        logger.info("Running complete data processing pipeline...")
        
        # Load data
        raw_data = self.load_multi_institutional_data()
        if raw_data.empty:
            raise ValueError("No data available for processing")
        
        # Extract features
        feature_data = self.extract_key_features(raw_data)
        
        # Preprocess
        X, y = self.preprocess_data(feature_data)
        
        # Store processed data
        self.processed_data = pd.concat([X, y], axis=1)
        
        logger.info("Data processing pipeline completed successfully!")
        return X, y

class HeartDiseasePredictor:
    """
    Advanced cardiac risk prediction system with ensemble methods and explainable AI
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.ensemble_model = None
        self.feature_names = []
        self.scaler = StandardScaler()
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Model performance tracking
        self.performance_metrics = {}
        self.feature_importance = {}
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
    
    def create_models(self, feature_names: List[str]):
        """Create ensemble of advanced ML models for cardiac risk prediction"""
        logger.info("Creating ensemble of ML models...")
        
        self.feature_names = feature_names
        
        # 1. Random Forest - robust for medical data
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        # 2. XGBoost - high performance gradient boosting
        try:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        except:
            logger.warning("XGBoost not available, skipping...")
        
        # 3. Gradient Boosting - another boosting approach
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # 4. Neural Network - for complex patterns
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=1000,
            random_state=42
        )
        
        # 5. Support Vector Machine - for non-linear patterns
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # 6. Logistic Regression - interpretable baseline
        self.models['logistic_regression'] = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        logger.info(f"Created {len(self.models)} ML models")
    
    def train_models(self, X: pd.DataFrame, y: pd.Series):
        """Train all models and create ensemble"""
        logger.info("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train individual models
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Store performance
                self.performance_metrics[name] = {
                    'accuracy': (y_pred == y_test).mean(),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted'),
                    'auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                logger.info(f"{name} trained successfully - Accuracy: {self.performance_metrics[name]['accuracy']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                self.performance_metrics[name] = {'error': str(e)}
        
        # Create ensemble
        self._create_ensemble(X_train, y_train)
        
        # Create explainability tools
        self._create_explainers(X_train)
        
        logger.info("Model training completed!")
    
    def _create_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Create voting ensemble from best models"""
        try:
            # Get best performing models
            best_models = []
            for name, metrics in self.performance_metrics.items():
                if 'error' not in metrics and metrics['accuracy'] > 0.7:
                    best_models.append((name, self.models[name]))
            
            if len(best_models) >= 2:
                # Create voting classifier
                estimators = [(name, model) for name, model in best_models]
                self.ensemble_model = VotingClassifier(
                    estimators=estimators,
                    voting='soft'
                )
                self.ensemble_model.fit(X_train, y_train)
                logger.info(f"Created ensemble with {len(best_models)} models")
            else:
                # Use best single model
                best_name = max(self.performance_metrics.items(), 
                              key=lambda x: x[1].get('accuracy', 0) if 'error' not in x[1] else 0)[0]
                self.ensemble_model = self.models[best_name]
                logger.info(f"Using best single model: {best_name}")
                
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            # Fallback to random forest
            self.ensemble_model = self.models['random_forest']
    
    def _create_explainers(self, X_train: pd.DataFrame):
        """Create SHAP and LIME explainers"""
        try:
            # SHAP explainer for tree-based models
            if hasattr(self.ensemble_model, 'estimators_'):
                # For ensemble, use first tree-based model
                base_model = self.ensemble_model.estimators_[0][1]
            else:
                base_model = self.ensemble_model
            
            if hasattr(base_model, 'feature_importances_'):
                self.shap_explainer = shap.TreeExplainer(base_model)
                logger.info("SHAP explainer created")
            
            # LIME explainer
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=self.feature_names,
                class_names=['No Heart Disease', 'Heart Disease'],
                mode='classification'
            )
            logger.info("LIME explainer created")
            
        except Exception as e:
            logger.error(f"Error creating explainers: {e}")
    
    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict cardiac risk for a patient"""
        try:
            # Prepare features
            features = []
            for feature in self.feature_names:
                if feature in patient_data:
                    features.append(patient_data[feature])
                else:
                    # Use median values for missing features
                    features.append(0)
            
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            if self.ensemble_model is not None:
                risk_prob = self.ensemble_model.predict_proba(features_array)[0]
                risk_score = risk_prob[1]  # Probability of heart disease
            else:
                # Fallback to random forest
                risk_prob = self.models['random_forest'].predict_proba(features_array)[0]
                risk_score = risk_prob[1]
            
            # Determine risk level
            risk_level = self._get_risk_level(risk_score)
            
            # Generate explanations
            explanations = self._generate_explanations(features_array, patient_data)
            
            return {
                "patient_id": patient_data.get("patient_id", "unknown"),
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "confidence": float(max(risk_prob)),
                "explanations": explanations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in risk prediction: {e}")
            raise
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score >= self.risk_thresholds['critical']:
            return "Critical"
        elif risk_score >= self.risk_thresholds['high']:
            return "High"
        elif risk_score >= self.risk_thresholds['medium']:
            return "Medium"
        elif risk_score >= self.risk_thresholds['low']:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_explanations(self, features_array: np.ndarray, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP and LIME explanations"""
        explanations = {}
        
        try:
            # SHAP explanations
            if self.shap_explainer is not None:
                shap_values = self.shap_explainer.shap_values(features_array)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                # Get top features
                feature_importance = dict(zip(self.feature_names, abs(shap_values[0])))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                explanations['shap'] = {
                    'top_features': top_features,
                    'feature_importance': feature_importance
                }
            
            # LIME explanations
            if self.lime_explainer is not None:
                lime_exp = self.lime_explainer.explain_instance(
                    features_array[0], 
                    self.ensemble_model.predict_proba,
                    num_features=5
                )
                
                explanations['lime'] = {
                    'explanation': str(lime_exp),
                    'feature_weights': lime_exp.as_list()
                }
                
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            explanations['error'] = str(e)
        
        return explanations
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive model performance summary"""
        return {
            'models_trained': len([m for m in self.performance_metrics.values() if 'error' not in m]),
            'ensemble_created': self.ensemble_model is not None,
            'best_model': max(self.performance_metrics.items(), 
                            key=lambda x: x[1].get('accuracy', 0) if 'error' not in x[1] else 0)[0],
            'performance_metrics': self.performance_metrics,
            'explainability_available': {
                'shap': self.shap_explainer is not None,
                'lime': self.lime_explainer is not None
            }
        }
    
    def save_models(self):
        """Save trained models"""
        try:
            # Save ensemble model
            if self.ensemble_model is not None:
                with open(self.model_dir / 'ensemble_model.pkl', 'wb') as f:
                    pickle.dump(self.ensemble_model, f)
            
            # Save individual models
            for name, model in self.models.items():
                with open(self.model_dir / f'{name}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
            
            # Save explainers
            if self.shap_explainer is not None:
                with open(self.model_dir / 'shap_explainer.pkl', 'wb') as f:
                    pickle.dump(self.shap_explainer, f)
            
            if self.lime_explainer is not None:
                with open(self.model_dir / 'lime_explainer.pkl', 'wb') as f:
                    pickle.dump(self.lime_explainer, f)
            
            logger.info("Models saved successfully!")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

class HeartDiseasePlatform:
    """
    Main platform orchestrator for the Heart Disease AI System
    """
    
    def __init__(self):
        self.data_processor = None
        self.risk_predictor = None
        self.is_running = False
        
    def initialize_system(self):
        """Initialize the complete system"""
        logger.info("🚀 Initializing Heart Disease AI Platform...")
        
        try:
            # Step 1: Initialize data processor
            logger.info("📊 Initializing data processor...")
            self.data_processor = HeartDiseaseDataProcessor()
            
            # Step 2: Process data pipeline
            logger.info("🔄 Processing multi-institutional data...")
            X, y = self.data_processor.process_pipeline()
            
            # Step 3: Initialize and train ML models
            logger.info("🤖 Initializing and training ML models...")
            self.risk_predictor = HeartDiseasePredictor()
            self.risk_predictor.create_models(X.columns.tolist())
            self.risk_predictor.train_models(X, y)
            
            # Step 4: Save models
            logger.info("💾 Saving trained models...")
            self.risk_predictor.save_models()
            
            # Step 5: Display system summary
            self.display_system_summary(X, y)
            
            logger.info("✅ System initialization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during system initialization: {e}")
            return False
    
    def display_system_summary(self, X: pd.DataFrame, y: pd.Series):
        """Display comprehensive system summary"""
        print("\n" + "="*80)
        print("🏥 HEART DISEASE AI PLATFORM - SYSTEM SUMMARY")
        print("="*80)
        
        # Data summary
        data_summary = self.data_processor.get_data_summary()
        print(f"\n📊 DATA SUMMARY:")
        print(f"   • Total Patients: {data_summary['total_patients']}")
        print(f"   • Total Features: {data_summary['total_features']}")
        print(f"   • Data Sources: {', '.join(data_summary['data_sources'])}")
        print(f"   • Target Distribution: {y.value_counts().to_dict()}")
        
        # Model performance
        performance_summary = self.risk_predictor.get_model_performance_summary()
        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"   • Models Trained: {performance_summary['models_trained']}")
        print(f"   • Ensemble Created: {performance_summary['ensemble_created']}")
        print(f"   • Best Model: {performance_summary['best_model']}")
        print(f"   • Explainability: SHAP={performance_summary['explainability_available']['shap']}, LIME={performance_summary['explainability_available']['lime']}")
        
        # Feature information
        print(f"\n🔍 FEATURE INFORMATION:")
        for feature in data_summary['feature_names']:
            desc = data_summary['feature_descriptions'].get(feature, 'No description')
            print(f"   • {feature}: {desc}")
        
        print("\n" + "="*80)
    
    def run_demo(self):
        """Run demonstration with sample patients"""
        logger.info("🎯 Running Heart Disease AI Platform Demo...")
        
        # Sample patients for demonstration
        sample_patients = [
            {
                "patient_id": "DEMO_001",
                "age": 65,
                "sex": 1,
                "cp": 4,
                "trestbps": 160,
                "chol": 250,
                "fbs": 1,
                "restecg": 2,
                "thalach": 140,
                "exang": 1,
                "oldpeak": 2.5,
                "slope": 2,
                "ca": 2,
                "thal": 6
            },
            {
                "patient_id": "DEMO_002", 
                "age": 45,
                "sex": 0,
                "cp": 1,
                "trestbps": 120,
                "chol": 180,
                "fbs": 0,
                "restecg": 0,
                "thalach": 170,
                "exang": 0,
                "oldpeak": 0.0,
                "slope": 1,
                "ca": 0,
                "thal": 3
            },
            {
                "patient_id": "DEMO_003",
                "age": 55,
                "sex": 1,
                "cp": 3,
                "trestbps": 140,
                "chol": 220,
                "fbs": 1,
                "restecg": 1,
                "thalach": 150,
                "exang": 1,
                "oldpeak": 1.5,
                "slope": 2,
                "ca": 1,
                "thal": 6
            }
        ]
        
        print("\n🎯 DEMONSTRATING RISK ASSESSMENT FOR SAMPLE PATIENTS:")
        print("="*80)
        
        for patient in sample_patients:
            try:
                result = self.risk_predictor.predict_risk(patient)
                
                print(f"\n👤 Patient: {result['patient_id']}")
                print(f"   Risk Score: {result['risk_score']:.3f}")
                print(f"   Risk Level: {result['risk_level']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                
                # Show top features from SHAP
                if 'shap' in result['explanations']:
                    print(f"   Top Risk Factors:")
                    for feature, importance in result['explanations']['shap']['top_features'][:3]:
                        print(f"     • {feature}: {importance:.3f}")
                
            except Exception as e:
                print(f"❌ Error assessing patient {patient['patient_id']}: {e}")
        
        print("\n" + "="*80)
        logger.info("Demo completed successfully!")

def main():
    """Main entry point"""
    print("🏥 Heart Disease AI Platform")
    print("="*50)
    
    # Initialize platform
    platform = HeartDiseasePlatform()
    
    # Initialize system
    if platform.initialize_system():
        # Run demo
        platform.run_demo()
    else:
        print("❌ Failed to initialize system. Please check logs for details.")

if __name__ == "__main__":
    main()
