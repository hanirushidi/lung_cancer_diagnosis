# Lung Cancer Diagnosis Prediction Application

A machine learning-based desktop application with a user-friendly GUI built using `tkinter`. This app helps assess the risk of lung cancer based on user inputs such as gender, age, smoking habits, and health symptoms.

## Features

- Intuitive multi-page GUI for user input and results
- Symptom-based lung cancer risk prediction
- Machine learning model integration
- Easy-to-use form for health information entry

## Prerequisites

- **Python 3.9 or higher** (recommended)
- Required Python libraries:
  - `tkinter` (built-in with Python)
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `numpy`

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/lung_cancer_diagnosis_app.git
cd lung_cancer_diagnosis_app
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, generate it by running:
```bash
pip freeze > requirements.txt
```

## Dataset Setup

- Place your `lung_cancer.csv` dataset in the `data/` directory.
- Ensure the dataset matches the format expected by the application, with columns like:
  - GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN, and LUNG_CANCER.

## Training the Model

Run the following command to train the model and save it:
```bash
python model/preprocessing.py
```

## Running the Application

Start the application with:
```bash
python app.py
```

## Usage Instructions

1. Open the application.
2. Fill in the health-related information.
3. Submit the form to get a risk prediction.
4. View the result and further insights.

## Project Structure
```
lung_cancer_diagnosis_app/
├── app.py                 # Main application file
├── model/
│   ├── preprocessing.py   # Model training script
│   └── lung_cancer_model.pkl  # Trained model file
├── pages/
│   ├── form_page.py       # User input form
│   └── result_page.py     # Result display page
├── data/
│   └── lung_cancer.csv    # Dataset file
├── requirements.txt       # List of dependencies
└── README.txt             # This file
```

## Requirements File
Ensure your `requirements.txt` contains:
```
pandas
scikit-learn
joblib
numpy
```

Generate it using:
```bash
pip freeze > requirements.txt
```

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments
- Dataset providers
- scikit-learn documentation
- Tkinter community

