from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model artifacts
model_data = joblib.load('model/lung_cancer_model.pkl')
model = model_data['model']
encoders = model_data['encoders']
feature_names = model_data['feature_names']
scaler = model_data['scaler']
symptom_features = model_data['symptom_features']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        # Process form data
        form_data = request.form.to_dict()
        
        # Create ordered dataframe
        input_df = pd.DataFrame([form_data])[feature_names]
        
        # Preprocess data
        for col in input_df.columns:
            if col in encoders:
                # Handle unseen values
                input_df[col] = input_df[col].apply(
                    lambda x: x if x in encoders[col].classes_ else 'NO'
                )
                input_df[col] = encoders[col].transform(input_df[col])
        
        # Scale age
        input_df['AGE'] = scaler.transform(input_df[['AGE']].values.astype(float))
        
        # Convert to numeric
        input_df = input_df.astype(float)
        
        # Predict
        probability = model.predict_proba(input_df)[0][1]
        prediction = probability >= 0.45  # Lower threshold for symptoms
        
        # Get symptom importances
        symptom_importance = {
            feat: model.feature_importances_[list(feature_names).index(feat)]
            for feat in symptom_features
        }
        top_indicators = dict(sorted(symptom_importance.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:4])
        
        # Generate description
        description = generate_medical_description(prediction, probability, top_indicators)
        
        return render_template('result.html',
                            prediction=prediction,
                            probability=probability,
                            indicators=top_indicators,
                            description=description)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('error.html'), 500

def generate_medical_description(prediction, probability, indicators):
    symptom_names = {
        'CHEST_PAIN': 'Chest Pain',
        'SHORTNESS_OF_BREATH': 'Shortness of Breath',
        'WHEEZING': 'Wheezing',
        'COUGHING': 'Persistent Cough'
    }
    
    base = "<div class='medical-report'>"
    base += f"<h3 class='risk-level'>{'High Risk' if prediction else 'Low Risk'}</h3>"
    base += f"<p>Cancer Probability: <strong>{probability:.1%}</strong></p>"
    
    if prediction:
        base += "<div class='critical-findings'>"
        base += "<h4>Critical Symptoms Detected:</h4><ul>"
        for symptom, impact in indicators.items():
            base += f"<li>{symptom_names.get(symptom, symptom)} " \
                   f"(Impact Score: {impact:.2f}/1.0)</li>"
        base += "</ul></div>"
        base += "<div class='recommendation'><p> <strong>Immediate Action Required:</strong></p>"
        base += "<ul><li>Consult pulmonologist within 48 hours</li>"
        base += "<li>Schedule CT scan and biopsy</li>"
        base += "<li>Monitor symptom progression daily</li></ul></div>"
    else:
        base += "<div class='findings'><p>No critical symptom patterns detected.</p>"
        base += "<p>Recommend regular screening for high-risk individuals.</p></div>"
    
    base += "</div>"
    return base

if __name__ == '__main__':
    app.run(debug=True)