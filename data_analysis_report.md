# 🧠 Healthcare Data Analysis Report: Stroke Prediction

This report provides a comprehensive analysis of the cleaned healthcare dataset, focusing on feature distributions, data classifications, and their relationship with stroke events.

---

## 1. Overview of the Dataset

| Metric | Value |
|------|------|
| Total Patient Records | 5,110 |
| Stroke Cases | 249 (4.87%) |
| Non-Stroke Cases | 4,861 (95.13%) |
| Average Age | 43.23 years |
| Average Glucose Level | 106.15 mg/dL |
| Average BMI | 28.92 kg/m² |

---

## 2. Feature Distributions

### 2.1 Numerical Distributions

The numerical features (Age, Glucose Level, and BMI) show varying distribution patterns:

- **Age**: Exhibits a fairly balanced distribution across different life stages.  
- **Glucose Level**: Shows a primary peak around 80–100 mg/dL, with a long tail indicating patients with hyperglycemia.  
- **BMI**: Follows a near-normal distribution centered around 29, which falls in the *overweight* category.

---

## 3. Data Classification and Stroke Analysis

### 3.1 Categorical Breakdown

The dataset is classified into various aspects such as Age Group, Smoking Habit, and Gender.

#### Age Group Distribution
- Middle-aged: 2,219 patients  
- Young: 1,515 patients  
- Senior: 1,376 patients  

#### Smoking Habit Classification
- Non-smoker: 1,892  
- Unknown: 1,544  
- Ex-smoker: 885  
- Current smoker: 789  

---

### 3.2 Stroke Incidence Across Categories

The analysis shows how stroke events vary across different demographic and lifestyle classifications.

**Key Insight:**  
Stroke incidence significantly increases in:
- Senior age group  
- Patients with high glucose levels  
- Individuals with a history of hypertension  

---

## 4. Correlation Analysis

A correlation heatmap reveals relationships between numeric variables:

- **Age** shows the strongest positive correlation with stroke events  
- Followed by **glucose level**  
- Then **hypertension status**

---

## 5. Conclusion

The dataset is highly imbalanced toward non-stroke cases (~95%), which is typical for medical screening data.

To build an effective prediction model, it is recommended to apply:
- Oversampling techniques (e.g., SMOTE)  
- Adjusted loss functions  
- Class balancing strategies  
