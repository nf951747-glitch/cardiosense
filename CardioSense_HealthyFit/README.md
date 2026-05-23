# CardioSense 🫀
### Heart Disease Risk Prediction System

CardioSense is a machine learning web application that predicts a user's risk of heart disease based on 16 clinical and lifestyle indicators. Users fill out a health form and receive an instant, personalised risk assessment powered by an ensemble of four trained ML models.

> ⚠️ **Medical Disclaimer:** CardioSense is an academic project built for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## Screenshots

Home Page 
<img width="1920" height="915" alt="image" src="https://github.com/user-attachments/assets/fc09c33e-1efa-4f96-85ef-26dee99a781e" />
<img width="1920" height="913" alt="image" src="https://github.com/user-attachments/assets/68cbb22b-f9c1-431f-be46-ad3b5043c7b9" />
<img width="1920" height="916" alt="image" src="https://github.com/user-attachments/assets/4d6eb7a3-e936-4171-874e-664afe525438" />


Prediction Form 
<img width="1920" height="915" alt="image" src="https://github.com/user-attachments/assets/411e020b-148e-45a1-b4b9-d598362bc698" />
<img width="1920" height="917" alt="image" src="https://github.com/user-attachments/assets/84acb7c1-14b4-4168-87fc-84e0e84dddee" />
<img width="1920" height="915" alt="image" src="https://github.com/user-attachments/assets/736b9599-734d-4503-9dfe-555bb61e39d9" />


Result Page 
<img width="1920" height="912" alt="image" src="https://github.com/user-attachments/assets/b80b7eab-aaef-4aa6-b423-b047ee884894" />
<img width="1920" height="918" alt="image" src="https://github.com/user-attachments/assets/d5d97fe4-5b94-431b-9b6a-ed0be3207e2b" />
<img width="1920" height="915" alt="image" src="https://github.com/user-attachments/assets/84cf3d91-8a69-410d-9087-d28f3a72526c" />
<img width="1920" height="920" alt="image" src="https://github.com/user-attachments/assets/e249e24f-bed7-432d-810b-e8efb177c03c" />


---
## Features

- **Multi-model ensemble prediction** — runs all 4 models simultaneously and uses smart majority voting
- **Real-time risk percentage** with Low / Moderate / High risk tiers
- **Personalised recommendations** based on user's specific health indicators (cholesterol, BMI, smoking, diabetes, blood pressure)
- **Model comparison panel** on the result page showing all 4 models' individual predictions and metrics
- **Tie-breaking logic** — when models split 2-2, the Voting Classifier acts as the final decider
- **Input validation** with clear error messages for every form field
- Clean 3-page UI: Home → Form → Result

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML & Data | Python, scikit-learn, pandas, NumPy |
| Data Preprocessing | KNN Imputer, StandardScaler |
| Web Backend | Flask |
| Frontend | HTML, CSS, Bootstrap |
| Model Serialisation | joblib |
| Production Server | Gunicorn |

---

## Dataset

- **Source:** Kaggle Heart Disease Dataset (UCI-based), expanded with synthetic samples
- **Final size:** 10,000 rows × 17 columns
- **Target variable:** `heart_disease` (0 = No, 1 = Yes)
- **Train / Test split:** 75% / 25% (stratified)

**16 input features used:**

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Gender (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = Yes) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia type (0–3) |
| `smoking` | Smoking status (1 = Yes) |
| `diabetes` | Diabetes status (1 = Yes) |
| `bmi` | Body Mass Index |

---

## ML Models & Results

Four ensemble models were trained and compared. All models achieved a **healthy fit** (train-test gap < 10%), confirmed after applying anti-overfitting regularisation.

### Test-Set Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ✅ | **65.3%** | 61.1% | **66.4%** | **63.6%** | **71.6%** |
| Gradient Boosting | 64.4% | **62.0%** | 57.1% | 59.4% | 69.0% |
| AdaBoost | 62.9% | 60.1% | 55.9% | 57.9% | 66.5% |
| Voting Classifier | 64.8% | 61.5% | 60.9% | 61.2% | 70.2% |

**Random Forest is the strongest model overall**, achieving the highest scores across accuracy, recall, F1, and ROC-AUC. Recall and ROC-AUC are treated as the primary indicators of clinical usefulness — missing an actual heart disease patient is far more costly than a false alarm. The **Voting Classifier** is used in deployment for its stable, balanced predictions across all input combinations.

### Overfitting Mitigation

A key challenge during development was reducing the train/test accuracy gap. Changes applied:

**Random Forest**
- Reduced `max_depth` from 10 → 6
- Increased `min_samples_leaf` from 5 → 10
- Added `min_samples_split=20`
- Feature subsampling with `max_features='sqrt'`

**Gradient Boosting**
- Reduced `n_estimators` from 200 → 150
- Lowered `learning_rate` from 0.1 → 0.05
- Added `subsample=0.8` for stochastic boosting
- Added `max_features='sqrt'`

**AdaBoost**
- Reduced `n_estimators` from 200 → 150

Result: all four models show a train-test gap well under 10%, confirming healthy generalisation.

---

## Project Structure

```
CardioSense_HealthyFit/
│
├── app.py                                   # Flask backend — routes & prediction logic
├── train_model.py                           # Model training script (run once)
├── heart_disease_ccp.ipynb                  # Full ML pipeline notebook
├── heart_disease_dataset.csv                # Dataset (Kaggle + synthetic)
├── README.md
├── requirements.txt
├── MODEL_CHANGES.txt                        # Overfitting fix log
│
├── model/
│   ├── heart_model.pkl                      # Voting Classifier (deployed model)
│   ├── rf_model.pkl                         # Random Forest
│   ├── gb_model.pkl                         # Gradient Boosting
│   ├── ada_model.pkl                        # AdaBoost
│   ├── scaler.pkl                           # StandardScaler
│   └── metrics.json                         # Stored test-set metrics for all 4 models
│
├── templates/
│   ├── index.html                           # Home page
│   ├── predict.html                         # Health form (16 inputs)
│   └── result.html                          # Risk result + model comparison
│
├── static/
│   └── css/
│       └── style.css
│
└── plots/
    ├── plot_step6_age_distribution.png
    ├── plot_step7_smoking_vs_heartdisease.png
    ├── plot_step8_boxplots.png
    ├── plot_step9_correlation_heatmap.png
    ├── plot_step10_gender_vs_heartdisease.png
    ├── plot_step21_confusion_matrices.png
    ├── plot_step23_model_comparison.png
    └── plot_step24_feature_importance.png
```

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/nf951747-glitch/cardiosense.git
cd cardiosense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models (only needed if .pkl files are missing)
python train_model.py

# 4. Start the Flask app
python app.py

# 5. Open in your browser
# http://127.0.0.1:5000
```

---

## How It Works

1. **User fills the form** on the `/predict` page with 16 health indicators
2. **Flask validates all inputs** and returns specific error messages if any field is missing or invalid
3. **All 4 models run simultaneously** on the scaled input
4. **Smart voting logic** picks the final result:
   - Clear majority (3-1 or 4-0) → majority wins
   - Tie (2-2) → Voting Classifier acts as tiebreaker
5. **Result page shows:**
   - Final prediction (High Risk / Low Risk) with risk percentage
   - Individual predictions from all 4 models
   - Personalised health recommendations based on the user's specific inputs

---

## Author

**Your Name**
BSCS Student — Lahore Garrison University
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/nf951747-glitch)

---

*Built as a Computer Science course project — 2025/2026*
