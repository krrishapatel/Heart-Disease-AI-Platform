#!/usr/bin/env python3
"""
Simple Demo Script for Heart Disease AI Platform
Shows how to use the platform for risk assessment
"""

from heart_disease_ai_platform import HeartDiseasePlatform
import json

def run_simple_demo():
    """Run a simple demonstration of the platform"""
    print("🏥 Heart Disease AI Platform - Simple Demo")
    print("=" * 60)
    
    try:
        # Initialize the platform
        print("🚀 Initializing platform...")
        platform = HeartDiseasePlatform()
        
        # Initialize system (this will train models)
        print("🤖 Training ML models (this may take a few minutes)...")
        if platform.initialize_system():
            print("✅ Platform initialized successfully!")
            
            # Demo patient data
            demo_patients = [
                {
                    "name": "High Risk Patient",
                    "data": {
                        "patient_id": "DEMO_HIGH_001",
                        "age": 70,
                        "sex": 1,
                        "cp": 4,
                        "trestbps": 180,
                        "chol": 300,
                        "fbs": 1,
                        "restecg": 2,
                        "thalach": 120,
                        "exang": 1,
                        "oldpeak": 3.0,
                        "slope": 2,
                        "ca": 3,
                        "thal": 6
                    }
                },
                {
                    "name": "Medium Risk Patient", 
                    "data": {
                        "patient_id": "DEMO_MED_001",
                        "age": 55,
                        "sex": 0,
                        "cp": 2,
                        "trestbps": 140,
                        "chol": 220,
                        "fbs": 0,
                        "restecg": 1,
                        "thalach": 150,
                        "exang": 0,
                        "oldpeak": 1.0,
                        "slope": 1,
                        "ca": 1,
                        "thal": 3
                    }
                },
                {
                    "name": "Low Risk Patient",
                    "data": {
                        "patient_id": "DEMO_LOW_001",
                        "age": 35,
                        "sex": 0,
                        "cp": 1,
                        "trestbps": 110,
                        "chol": 160,
                        "fbs": 0,
                        "restecg": 0,
                        "thalach": 180,
                        "exang": 0,
                        "oldpeak": 0.0,
                        "slope": 1,
                        "ca": 0,
                        "thal": 3
                    }
                }
            ]
            
            print("\n🎯 Running Risk Assessment Demo...")
            print("=" * 60)
            
            for patient_info in demo_patients:
                print(f"\n👤 {patient_info['name']}")
                print("-" * 40)
                
                try:
                    # Get risk assessment
                    result = platform.risk_predictor.predict_risk(patient_info['data'])
                    
                    # Display results
                    print(f"Patient ID: {result['patient_id']}")
                    print(f"Risk Score: {result['risk_score']:.3f}")
                    print(f"Risk Level: {result['risk_level']}")
                    print(f"Confidence: {result['confidence']:.3f}")
                    
                    # Show top risk factors
                    if 'shap' in result['explanations']:
                        print(f"\nTop Risk Factors:")
                        for feature, importance in result['explanations']['shap']['top_features'][:3]:
                            print(f"  • {feature}: {importance:.3f}")
                    
                    # Color code risk level
                    risk_colors = {
                        "Critical": "🔴",
                        "High": "🟠", 
                        "Medium": "🟡",
                        "Low": "🟢",
                        "Very Low": "🔵"
                    }
                    print(f"\nRisk Assessment: {risk_colors.get(result['risk_level'], '⚪')} {result['risk_level']}")
                    
                except Exception as e:
                    print(f"❌ Error assessing patient: {e}")
                
                print("-" * 40)
            
            # Show system summary
            print("\n📊 System Summary")
            print("=" * 60)
            
            # Data summary
            data_summary = platform.data_processor.get_data_summary()
            print(f"Total Patients: {data_summary['total_patients']}")
            print(f"Total Features: {data_summary['total_features']}")
            print(f"Data Sources: {', '.join(data_summary['data_sources'])}")
            
            # Model performance
            perf_summary = platform.risk_predictor.get_model_performance_summary()
            print(f"\nModels Trained: {perf_summary['models_trained']}")
            print(f"Ensemble Created: {perf_summary['ensemble_created']}")
            print(f"Best Model: {perf_summary['best_model']}")
            
            print("\n✅ Demo completed successfully!")
            
        else:
            print("❌ Failed to initialize platform")
            
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements_github.txt")

def interactive_demo():
    """Interactive demo where user can input patient data"""
    print("\n🎮 Interactive Demo Mode")
    print("=" * 40)
    
    try:
        # Initialize platform
        platform = HeartDiseasePlatform()
        if not platform.initialize_system():
            print("❌ Failed to initialize platform for interactive demo")
            return
        
        print("✅ Platform ready for interactive use!")
        print("\nEnter patient data (press Enter to use default values):")
        
        # Get patient data interactively
        patient_data = {}
        
        # Define feature prompts
        features = {
            'age': ('Age (years)', 50),
            'sex': ('Sex (1=male, 0=female)', 1),
            'cp': ('Chest pain type (1-4)', 2),
            'trestbps': ('Resting blood pressure (mm Hg)', 140),
            'chol': ('Cholesterol (mg/dl)', 200),
            'fbs': ('Fasting blood sugar > 120 (1=yes, 0=no)', 0),
            'restecg': ('Resting ECG (0-2)', 1),
            'thalach': ('Max heart rate', 150),
            'exang': ('Exercise angina (1=yes, 0=no)', 0),
            'oldpeak': ('ST depression', 1.0),
            'slope': ('Slope of ST segment (1-3)', 1),
            'ca': ('Number of vessels (0-3)', 0),
            'thal': ('Thalassemia (3,6,7)', 3)
        }
        
        for feature, (description, default) in features.items():
            user_input = input(f"{description} [{default}]: ").strip()
            if user_input:
                try:
                    patient_data[feature] = float(user_input)
                except ValueError:
                    print(f"Invalid input, using default: {default}")
                    patient_data[feature] = default
            else:
                patient_data[feature] = default
        
        # Add patient ID
        patient_data['patient_id'] = 'INTERACTIVE_001'
        
        print(f"\n📋 Patient Data Summary:")
        for feature, value in patient_data.items():
            if feature != 'patient_id':
                print(f"  {feature}: {value}")
        
        # Get risk assessment
        print(f"\n🔍 Assessing Risk...")
        result = platform.risk_predictor.predict_risk(patient_data)
        
        # Display results
        print(f"\n📊 Risk Assessment Results:")
        print(f"Risk Score: {result['risk_score']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        # Show explanations
        if 'shap' in result['explanations']:
            print(f"\n🔍 Top Risk Factors:")
            for feature, importance in result['explanations']['shap']['top_features'][:5]:
                print(f"  • {feature}: {importance:.3f}")
        
        print(f"\n✅ Interactive assessment completed!")
        
    except Exception as e:
        print(f"❌ Error in interactive demo: {e}")

if __name__ == "__main__":
    print("🏥 Heart Disease AI Platform - Demo Suite")
    print("=" * 50)
    
    while True:
        print("\nChoose demo mode:")
        print("1. Simple Demo (predefined patients)")
        print("2. Interactive Demo (input your own data)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            run_simple_demo()
        elif choice == '2':
            interactive_demo()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
        if choice in ['1', '2']:
            input("\nPress Enter to continue...")
