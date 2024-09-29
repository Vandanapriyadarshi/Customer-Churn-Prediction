# Customer-Churn-Prediction
Here's a basic outline of the code for a Customer Churn Prediction model using the Telco Customer Churn dataset from Kaggle. This project will involve:

Data Loading
Data Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Model Building
Model Evaluation
Below is the code to guide you through the process:

1. Install Required Libraries:
Make sure to install necessary libraries if you haven't done so:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
2. Load the Data:
python
Copy code
import pandas as pd

# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Display the first few rows of the data
df.head()
3. Data Preprocessing:
Check for missing values and handle them.
Convert categorical columns to numerical.
Drop irrelevant columns.
python
Copy code
# Check for missing values
print(df.isnull().sum())

# Drop 'customerID' as it's not relevant for prediction
df = df.drop(['customerID'], axis=1)

# Convert 'TotalCharges' to numeric and handle errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing 'TotalCharges' with median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

df.head()  # Check the cleaned data
4. Exploratory Data Analysis (EDA):
You can visualize the distribution of churners vs non-churners and important features.


import matplotlib.pyplot as plt
import seaborn as sns

# Countplot of Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn_Yes', data=df)
plt.title('Churn Distribution')
plt.show()

# Correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()
5. Split the Data into Train/Test Sets:
python
Copy code
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
6. Model Building:
We'll use Logistic Regression as a simple baseline model. You can experiment with other models such as Decision Trees, Random Forest, etc.


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
7. Model Evaluation:
In addition to accuracy, you may want to look at other metrics such as precision, recall, and F1-score to better understand the model’s performance.


# Precision, Recall, and F1-score
print("Classification Report:\n", classification_report(y_test, y_pred))
8. Feature Importance:
Logistic Regression allows you to check which features were the most important in predicting churn.


import numpy as np

# Get feature importance from Logistic Regression coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
coefficients = coefficients.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=coefficients)
plt.title('Feature Importance')
plt.show()
9. Advanced Models (Optional):
You can also use more complex models such as Random Forest or XGBoost to improve accuracy.

Example using Random Forest:

from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_y_pred))
Conclusion:
This code will guide you through building a customer churn prediction model, from data cleaning to model building and evaluation. You can further improve it by tuning hyperparameters, applying cross-validation, or using ensemble methods like XGBoost.
