# Master's thesis - Cardiorespiratory Fitness Prediction (VO₂max) 

### 📖 Project Overview
This repository contains the complete code, datasets, and Streamlit application for the Master's Thesis project focused on predicting cardiorespiratory fitness (VO₂max) using non-exercise variables from NHANES datasets.

The detailed report of this Master's Thesis is available in the PDF file included in this repository.  

### ⚙️ Usage Instructions

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```
   
2. **Configure Environment Variables**

Copy .env.example to .env and modify the paths or credentials if needed:

  ```bash
  cp .env.example .env
  ```

3. **Run the Streamlit Application**

  ```bash
  streamlit run app/app.py
  ```
This will launch the interactive VO₂max prediction app in your web browser. Enter the requested biometric data (age, sex, BMI, body fat %, education level, smoker status, ethnicity) to obtain an estimated VO₂max and interpretation.

4. **Explore Jupyter Notebooks**

- `1_ETL.ipynb`: Data extraction, transformation, and loading workflow.

- `2_EDA_and_Clustering.ipynb`: Exploratory Data Analysis and clustering.

- `3_ML_models.ipynb`: Training, evaluation, and hyperparameter tuning of machine learning models.
