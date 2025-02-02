import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
import joblib

# Load dataset
df = pd.read_csv('data/lung_cancer.csv')

# Feature engineering: Boost key symptoms
key_symptoms = ['CHEST_PAIN', 'SHORTNESS_OF_BREATH', 'WHEEZING', 'COUGHING']
symptom_mask = df[key_symptoms].apply(lambda x: x == 'YES').any(axis=1)
df = pd.concat([df, df[symptom_mask].sample(frac=0.7, replace=True)], axis=0)

# Preprocessing
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Reduce age impact
scaler = StandardScaler()
df['AGE'] = scaler.fit_transform(df[['AGE']])

# Split data
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature weighting
feature_weights = {
    'AGE': 0.4,
    'CHEST_PAIN': 3.0,
    'SHORTNESS_OF_BREATH': 3.5,
    'WHEEZING': 2.5,
    'COUGHING': 2.5
}

# Apply sample weights
sample_weights = np.ones(len(X_train))
for feature, weight in feature_weights.items():
    if feature in X_train.columns:
        sample_weights *= np.where(X_train[feature] == 1, weight, 1)

# Handle class imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight=class_weight_dict,
    max_depth=8,
    min_samples_split=5
)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Save artifacts
joblib.dump({
    'model': model,
    'encoders': label_encoders,
    'feature_names': list(X.columns),
    'scaler': scaler,
    'symptom_features': key_symptoms
}, 'model/lung_cancer_model.pkl')

print("Model trained with symptom-focused weights!")