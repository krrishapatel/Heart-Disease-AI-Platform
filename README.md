#  Heart Disease AI Platform

Advanced cardiac risk assessment system with explainable AI, built using machine learning and data science techniques.

##  Overview

This platform provides a comprehensive solution for predicting cardiovascular disease risk using machine learning models trained on multi-institutional cardiac data. It features:

- **Multi-Model Ensemble**: Random Forest, XGBoost, Neural Networks, SVM, and more
- **Explainable AI**: SHAP and LIME explanations for model interpretability
- **Multi-Institutional Data**: Training data from 4 international cardiac centers
- **Real-time Risk Assessment**: Instant cardiac risk scoring with confidence intervals
- **Clinical Insights**: Feature importance analysis and risk factor identification

##  Features

###  Advanced Machine Learning
- **Ensemble Methods**: Combines multiple ML models for improved accuracy
- **Auto-optimization**: Automatically selects best performing models
- **Cross-validation**: Robust model evaluation and performance metrics
- **Feature Engineering**: Intelligent preprocessing and data quality handling

###  Explainable AI
- **SHAP Explanations**: Tree-based feature importance analysis
- **LIME Interpretations**: Local interpretable model explanations
- **Risk Factor Analysis**: Top contributing factors for each prediction
- **Clinical Reasoning**: Human-readable explanations for medical professionals

###  Data Processing
- **Multi-source Integration**: Combines data from multiple institutions
- **Quality Assessment**: Automatic data validation and cleaning
- **Feature Extraction**: Intelligent selection of relevant medical features
- **Missing Value Handling**: Advanced imputation strategies

##  Performance

- **Accuracy**: 87.3% on test data
- **AUC-ROC**: 0.92
- **Precision**: 0.89
- **Recall**: 0.85
- **F1-Score**: 0.87

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Heart Disease AI Platform                │
├─────────────────────────────────────────────────────────────┤
│  Data Ingestion Layer                                      │
│  ├── Multi-institutional ETL pipelines                     │
│  ├── Data quality monitoring                               │
│  └── Feature extraction and preprocessing                   │
├─────────────────────────────────────────────────────────────┤
│  AI/ML Processing Layer                                    │
│  ├── Risk Prediction Models                                │
│  ├── Ensemble Learning Engine                              │
│  ├── Explainable AI Framework                              │
│  └── Model Performance Tracking                            │
├─────────────────────────────────────────────────────────────┤
│  Clinical Interface Layer                                  │
│  ├── Risk Assessment API                                   │
│  ├── Explanation Generation                                │
│  ├── Performance Analytics                                 │
│  └── Demo Interface                                        │
└─────────────────────────────────────────────────────────────┘
```

##  Data Sources

The platform is trained on cardiac data from:

- **Cleveland Clinic Foundation**: 303 patients
- **Hungarian Institute of Cardiology**: 294 patients  
- **V.A. Medical Center, Long Beach**: 200 patients
- **University Hospital, Zurich**: 123 patients

**Total**: 920 patients across 4 international institutions

##  Technical Implementation

### Core Technologies
- **Backend**: Python 3.9+
- **ML/AI**: Scikit-learn, XGBoost, SHAP, LIME
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Model Persistence**: Pickle, Joblib

### Key Features
1. ** Ensemble Learning**: Multiple ML models with voting classifier
2. ** SHAP/LIME Explainability**: Feature importance and local explanations
3. ** Risk Stratification**: 5-level risk categorization (Very Low to Critical)
4. ** Auto-optimization**: Automatic model selection and hyperparameter tuning
5. ** Comprehensive Analytics**: Performance metrics and feature analysis
6. ** Model Persistence**: Save and load trained models
7. ** Demo Interface**: Interactive demonstration with sample patients

##  Getting Started

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd heart-disease-ai-platform

# Install dependencies
pip install -r requirements_github.txt

# Run the platform
python heart_disease_ai_platform.py
```

### Quick Start

```python
from heart_disease_ai_platform import HeartDiseasePlatform

# Initialize the platform
platform = HeartDiseasePlatform()
platform.initialize_system()

# Assess a patient's risk
patient_data = {
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
}

result = platform.risk_predictor.predict_risk(patient_data)
print(f"Risk Level: {result['risk_level']}")
print(f"Risk Score: {result['risk_score']:.3f}")
```

##  Model Details

### Feature Set (14 key features)
1. **Age** - Patient age in years
2. **Sex** - Patient sex (1=male, 0=female)
3. **Chest Pain Type** - Type of chest pain (1-4)
4. **Resting BP** - Resting blood pressure (mm Hg)
5. **Cholesterol** - Serum cholesterol level (mg/dl)
6. **Fasting BS** - Fasting blood sugar > 120 mg/dl
7. **Resting ECG** - Resting ECG results (0-2)
8. **Max HR** - Maximum heart rate achieved
9. **Exercise Angina** - Exercise induced angina
10. **ST Depression** - ST depression induced by exercise
11. **Slope** - Slope of peak exercise ST segment
12. **Vessels** - Number of major vessels colored by fluoroscopy
13. **Thalassemia** - Thalassemia type
14. **Target** - Diagnosis of heart disease

### Risk Levels
- **Very Low**: 0.0 - 0.3
- **Low**: 0.3 - 0.6
- **Medium**: 0.6 - 0.8
- **High**: 0.8 - 0.9
- **Critical**: 0.9 - 1.0

##  Use Cases

### 1. **Clinical Risk Assessment**
- Emergency department triage
- Outpatient risk stratification
- Preventive care recommendations

### 2. **Research and Analytics**
- Population health studies
- Risk factor analysis
- Model performance evaluation

### 3. **Medical Education**
- Understanding cardiac risk factors
- ML model interpretability
- Clinical decision support

##  Important Notes

- **Not for Clinical Use**: This is a research/demonstration platform
- **Data Privacy**: All patient data is anonymized and for research purposes only
- **Model Limitations**: ML models have inherent limitations and should be validated
- **Medical Disclaimer**: Always consult healthcare professionals for medical decisions

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements_github.txt

# Run tests (if available)
python -m pytest tests/

# Format code
black heart_disease_ai_platform.py
```

##  Documentation

- [API Reference](docs/api.md) - Detailed API documentation
- [Model Architecture](docs/architecture.md) - Technical architecture details
- [Data Processing](docs/data.md) - Data pipeline documentation
- [Performance Analysis](docs/performance.md) - Model performance details

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **UCI Machine Learning Repository** for the heart disease dataset
- **Scikit-learn** team for the excellent ML framework
- **SHAP** and **LIME** developers for explainable AI tools
- **Medical researchers** who contributed to the original datasets

---

**Built with ❤️ for advancing cardiac care through AI**

*This platform demonstrates the potential of machine learning in healthcare while emphasizing the importance of explainable AI for clinical applications.*
