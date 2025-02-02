import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Create dictionary with realistic distributions
data = {
    # Demographic features
    'GENDER': np.random.choice(['M', 'F'], n_samples, p=[0.55, 0.45]),
    'AGE': np.clip(np.random.normal(60, 12, n_samples).astype(int), 20, 90),
    
    # Behavioral factors
    'SMOKING': np.random.choice(['YES', 'NO'], n_samples, p=[0.4, 0.6]),
    'ALCOHOL_CONSUMING': np.random.choice(['YES', 'NO'], n_samples, p=[0.35, 0.65]),
    'PEER_PRESSURE': np.random.choice(['YES', 'NO'], n_samples, p=[0.3, 0.7]),
    
    # Symptoms
    'YELLOW_FINGERS': np.random.choice(['YES', 'NO'], n_samples, p=[0.25, 0.75]),
    'COUGHING': np.random.choice(['YES', 'NO'], n_samples, p=[0.45, 0.55]),
    'SHORTNESS_OF_BREATH': np.random.choice(['YES', 'NO'], n_samples, p=[0.4, 0.6]),
    'SWALLOWING_DIFFICULTY': np.random.choice(['YES', 'NO'], n_samples, p=[0.2, 0.8]),
    'CHEST_PAIN': np.random.choice(['YES', 'NO'], n_samples, p=[0.35, 0.65]),
    
    # Medical history
    'ANXIETY': np.random.choice(['YES', 'NO'], n_samples, p=[0.3, 0.7]),
    'CHRONIC_DISEASE': np.random.choice(['YES', 'NO'], n_samples, p=[0.25, 0.75]),
    'ALLERGY': np.random.choice(['YES', 'NO'], n_samples, p=[0.3, 0.7]),
    
    # Physical signs
    'FATIGUE': np.random.choice(['YES', 'NO'], n_samples, p=[0.4, 0.6]),
    'WHEEZING': np.random.choice(['YES', 'NO'], n_samples, p=[0.3, 0.7]),
    
    # Target variable
    'LUNG_CANCER': np.random.choice(['YES', 'NO'], n_samples, p=[0.35, 0.65])
}

# Create DataFrame
df = pd.DataFrame(data)

# Add realistic correlations
def add_correlations(df):
    # Smoking correlates with yellow fingers and cancer
    df.loc[df['SMOKING'] == 'YES', 'YELLOW_FINGERS'] = np.random.choice(['YES', 'NO'], 
        size=sum(df['SMOKING'] == 'YES'), p=[0.7, 0.3])
    
    # Age correlation with cancer
    df.loc[df['AGE'] > 60, 'LUNG_CANCER'] = np.random.choice(['YES', 'NO'], 
        size=sum(df['AGE'] > 60), p=[0.5, 0.5])
    
    # Symptom clusters for cancer patients
    cancer_mask = df['LUNG_CANCER'] == 'YES'
    df.loc[cancer_mask, 'COUGHING'] = np.random.choice(['YES', 'NO'], size=sum(cancer_mask), p=[0.8, 0.2])
    df.loc[cancer_mask, 'CHEST_PAIN'] = np.random.choice(['YES', 'NO'], size=sum(cancer_mask), p=[0.7, 0.3])
    df.loc[cancer_mask, 'SHORTNESS_OF_BREATH'] = np.random.choice(['YES', 'NO'], size=sum(cancer_mask), p=[0.75, 0.25])
    
    return df

df = add_correlations(df)

# Save to CSV
df.to_csv('lung_cancer_dataset.csv', index=False)

print("Dataset created with shape:", df.shape)