#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Print versions to confirm installation
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Matplotlib version:", plt.__version__)
print("Seaborn version:", sns.__version__)
print("Scikit-learn version:", datasets.__version__)


# In[4]:


import pandas as pd

# Load the dataset
df = pd.read_csv('C:/Users/Lenovo/Downloads/archive/Telco-Customer-Churn.csv')

# Display the first few rows of the data
df.head()


# In[5]:


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


# In[6]:


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


# In[8]:


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# In[9]:


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


# In[10]:


# Precision, Recall, and F1-score
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[11]:


import numpy as np

# Get feature importance from Logistic Regression coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
coefficients = coefficients.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=coefficients)
plt.title('Feature Importance')
plt.show()


# In[12]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_y_pred))


# In[ ]:




