# 🏠 Real Estate Price Prediction

This project is a machine learning pipeline for predicting real estate prices in USD.  
The task was completed as part of a competition assignment.  

---

## 📊 Dataset
- The dataset contains apartment listings from Bishkek.  
- **Target:** `usd_price` (price in USD).  
- **Features include:**
  - **Categorical:** building type, condition, heating, etc.  
  - **Numeric:** square meters, floor ratio, latitude, longitude, etc.  

---

## 🛠 Data Preprocessing
1. **Target clipping** – removed outliers (kept values between 1st and 99th quantile).  
2. **Log-transform target** – applied `log1p(y)` to stabilize variance and reduce skew.  
3. **Feature processing:**
   - Categorical → one-hot encoding  
   - Numeric → median imputation + scaling  
   - Skewed numeric → log-transform + scaling  
4. All transformations are wrapped into a **ColumnTransformer** inside a scikit-learn **Pipeline**.  

---

## 🤖 Models
I trained several models using **cross-validation (KFold, 5 folds)** and evaluated with **MAPE (Mean Absolute Percentage Error)**:

- Linear Regression  
- Ridge Regression ✅ **Best so far (MAPE ≈ 0.103)**  
- Lasso Regression  
- ElasticNet  

📌 Planned extensions:  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

---

## 📈 Results

| Model            | CV MAPE (log target) |
|------------------|-----------------------|
| Linear Regression| 0.115                 |
| Ridge Regression | **0.103**             |
| Lasso Regression | 0.122                 |
| ElasticNet       | 0.119                 |
| Decision Tree    | _to be added_         |
| Random Forest    | _to be added_         |
| Gradient Boosting| _to be added_         |
<img width="601" height="455" alt="image" src="https://github.com/user-attachments/assets/2ae664f3-f16a-4bc3-be9a-c6454250a6e6" />

---

## 🔑 Feature Importance
- **Linear models:** coefficient analysis shows strong impact from `Площадь (area)` and `region`(address).  
- **Tree models:** feature importance plots will be added once trained.  

---

## ⚠️ Diagnostics
- Residual plots used to check heteroskedasticity.  
- Outliers clipped at **0.01–0.99 quantiles** of target distribution.  

---

## 💾 Model Saving
Each trained model is wrapped into a **Pipeline** (preprocessing + estimator) and saved with `joblib`:

```python
import joblib
joblib.dump(best_model, "ridge_pipeline.pkl")
