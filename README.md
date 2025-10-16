# Logistic Regression Project: Predicting Ad Clicks

In this project, we implement Logistic Regression algorithm with Python to predict whether an internet user will click on an advertisement. We build a binary classification model using the **Advertising Click** dataset, which contains various user features and their interaction with ads.

## Table of Contents

1. Introduction to Logistic Regression
2. Problem Statement
3. Data Overview
4. Data Preprocessing
5. Exploratory Data Analysis
6. Model Building
7. Model Evaluation
8. Results and Conclusion
9. References

## 1. Introduction to Logistic Regression

**Logistic Regression** is a fundamental classification algorithm in machine learning used for binary classification problems. Despite its name containing "regression," it is primarily employed for classification tasks where the target variable is categorical.

### Key Concepts:
- **Sigmoid Function**: Maps any real value to a probability between 0 and 1
- **Decision Boundary**: Threshold (typically 0.5) that separates classes
- **Cost Function**: Cross-entropy loss instead of mean squared error
- **Probability Output**: Returns probabilities that can be interpreted as confidence scores

## 2. Problem Statement

**Business Question**: Can we predict whether a user will click on an advertisement based on their demographic and behavioral characteristics?

**Objective**: Build a binary classifier using Logistic Regression to predict ad clicks (`Clicked on Ad = 0 or 1`) based on user features like time spent on site, age, income, and internet usage patterns.

## 3. Data Overview

The dataset contains the following features:

- **'Daily Time Spent on Site'**: Consumer time on site in minutes (continuous)
- **'Age'**: Customer age in years (continuous)
- **'Area Income'**: Average Income of geographical area of consumer (continuous)
- **'Daily Internet Usage'**: Average minutes per day consumer is on the internet (continuous)
- **'Ad Topic Line'**: Headline of the advertisement (categorical)
- **'City'**: City of consumer (categorical)
- **'Male'**: Whether consumer was male (binary: 0 or 1)
- **'Country'**: Country of consumer (categorical)
- **'Timestamp'**: Time when consumer clicked on Ad or closed window (datetime)
- **'Clicked on Ad'**: Target variable (binary: 0 or 1)

## 4. Data Preprocessing

### Steps Involved:
1. **Handling Missing Values**: Identify and treat missing data
2. **Feature Engineering**:
   - Extract time-based features from timestamp (hour, day of week, month)
   - Create relevant aggregates if needed
3. **Categorical Encoding**:
   - One-hot encoding for 'Country', 'City', 'Ad Topic Line'
   - Label encoding for ordinal categories if any
4. **Feature Scaling**: Standardize/Normalize numerical features
5. **Train-Test Split**: Split data into training and testing sets (typically 70-30 or 80-20)

## 5. Exploratory Data Analysis

### Key Analysis Areas:
1. **Target Variable Distribution**: Balance check of clicked vs non-clicked ads
2. **Correlation Analysis**: Relationship between features and target variable
3. **Feature Distributions**: 
   - Age distribution by click status
   - Time spent on site vs click rate
   - Income levels and click behavior
   - Internet usage patterns
4. **Categorical Analysis**:
   - Click rates by country
   - Gender differences in click behavior
   - Ad topic performance

### Expected Visualizations:
- Correlation heatmap
- Distribution plots for numerical features
- Count plots for categorical features
- Box plots showing feature distributions by click status

## 6. Model Building

### Logistic Regression Implementation:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Preprocessing
X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model training
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predictions
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
```

### Model Variations:
1. **Basic Logistic Regression**: With default parameters
2. **Regularized Models**: L1 (Lasso) and L2 (Ridge) regularization
3. **Hyperparameter Tuning**: Using GridSearchCV for optimal parameters

## 7. Model Evaluation

### Evaluation Metrics:
1. **Accuracy**: Overall correctness of predictions
2. **Precision**: Quality of positive predictions
3. **Recall**: Coverage of actual positive cases
4. **F1-Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Area under ROC curve measuring separability
6. **Confusion Matrix**: Detailed breakdown of predictions

### Key Evaluation Steps:
1. **Baseline Performance**: Compare against random/majority class classifier
2. **Cross-Validation**: Ensure model generalizability
3. **Feature Importance**: Identify most influential predictors
4. **Learning Curves**: Check for overfitting/underfitting

## 8. Results and Conclusion

### Expected Findings:
1. **Model Performance**: 
   - Expected accuracy: > 80% (given the nature of advertising data)
   - Key drivers: Time spent on site, daily internet usage, age likely to be strong predictors
   
2. **Business Insights**:
   - Demographic segments most likely to click ads
   - Optimal time for ad displays
   - Behavioral patterns of engaged users

3. **Model Limitations**:
   - Potential missing features (user interests, device type, etc.)
   - Temporal changes in user behavior
   - Privacy considerations in feature usage

### Conclusion:
The Logistic Regression model provides a interpretable and efficient solution for predicting ad clicks. The coefficients offer direct insights into feature importance, making it valuable for marketing strategy decisions.

## 9. References

1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning
2. Scikit-learn Documentation: Logistic Regression
3. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression
4. Google Analytics documentation for digital advertising metrics
5. Industry reports on digital advertising click-through rates

---

*This project demonstrates the practical application of Logistic Regression in digital marketing analytics, providing actionable insights for advertising optimization and user engagement strategies.*
