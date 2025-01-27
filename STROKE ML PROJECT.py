#!/usr/bin/env python
# coding: utf-8

# In[25]:


#All imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import preprocessing


# # I) DATA EXPLORATION

# In[26]:


import pandas as pd

# Load the dataset
file_path = 'stroke_data.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
data_info = {
    "Head": data.head(),
    "Shape": data.shape,
    "Columns": data.columns.tolist(),
    "Missing Values": data.isnull().sum(),
    "Data Types": data.dtypes
}

data_info


# Basic Overview of the Raw Dataset:
#     
# 1)Shape:
# The dataset contains 5,110 rows and 12 columns.
# 
# 2)Sample Data:
# A quick look at the first five rows shows features such as gender, age, hypertension, heart_disease, bmi, and the target variable stroke.
# 
# 3)MISSING VALUES:
# The bmi column has 201 missing values, which we need to address later.
# 

# In[27]:


data.head()


# In[36]:


#Evaluation the uniqueness of the id, to see if we do have any patient present with multiple row data 
data['id'].nunique()


# In[38]:


data ['gender'].unique()


# In[40]:


data[data['gender'] == 'Other']


# In[46]:


#Let's see the percentage of male and female in the dataset


# In[45]:


gender_percentage = data['gender'].value_counts(normalize = True)*100
gender_percentage

# VISUALIZING DISTRIBUTIONS
# In[10]:


# Distribution of numerical features
def plot_numerical_distributions(data, columns):
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], kde=True, bins=30, color='blue')
        plt.title(f"Distribution of {col}", fontsize=16)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
plot_numerical_distributions(data, ['age', 'avg_glucose_level', 'bmi'])


# Numerical Feature Distributions:
# 
# 1) Age:
# The distribution is slightly right-skewed.
# Most patients are between 0 and 80 years old, with a few outliers above 80.
# 
# 2)Average Glucose Level:
# A wide range of values, with some extreme outliers at the higher end.
# Indicates significant variation in glucose levels.
# 
# 3) BMI:
# The distribution is approximately normal, but there are missing values (gaps in the histogram)--> We'll handle these missing data later on

# In[50]:


# Function to summarize statistics for each column
def summarize_data(data):
    summary = []
    for col in data.columns:
        col_data = data[col]
        unique_values = col_data.nunique()
        col_type = col_data.dtype
        if col_type in ['int64', 'float64']:  # Numeric columns
            mean_val = col_data.mean()
            median_val = col_data.median()
            summary.append({
                "Column": col,
                "Type": "Numeric",
                "Unique Values": unique_values,
                "Mean": mean_val,
                "Median": median_val,
                "Categories/Values": "N/A"
            })
        else:  # Categorical or object columns
            categories = col_data.unique()
            summary.append({
                "Column": col,
                "Type": "Categorical",
                "Unique Values": unique_values,
                "Mean": "N/A",
                "Median": "N/A",
                "Categories/Values": categories
            })
    return pd.DataFrame(summary)

# Summarize the dataset 
data_summary_updated = summarize_data(data)
data_summary_updated

Now ,Let’s proceed with the analysis and visualization of categorical features.

This step will include:

Distribution plots for each categorical column (gender, ever_married, work_type, etc.).
Breakdown of the stroke variable across these categories to identify patterns.
# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set general plot style
sns.set(style="whitegrid")

# Function to plot distributions of categorical features
def plot_categorical_distributions(data, categorical_columns):
    for col in categorical_columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data, x=col, palette="viridis", order=data[col].value_counts().index)
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.ylabel("Count")
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.show()

# Categorical columns to analyze
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Plot distributions
plot_categorical_distributions(data, categorical_columns)


# Observations on Categorical Features:
# 
# 1-Gender:
# Most patients are either "Male" or "Female," with very few labeled as "Other." (actually only one is labeled "Other")
# 
# 2-Ever Married:
# Majority of patients are married, with "Yes" significantly outweighing "No."
# 
# 3-Work Type:
# Most patients are in "Private" jobs, followed by "Self-employed."
# 
# 4-Very few are "Never worked" or "Children."
# 
# 5-Residence Type:
# Fairly balanced between "Urban" and "Rural."
# 
# 6-Smoking Status:
# A significant portion of data falls into the "Unknown" category, which may need special handling.

# In[53]:


# Visualizing the distribution of 'hypertension' and 'heart_disease'
def plot_binary_distributions(data, binary_columns):
    for col in binary_columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data, x=col, palette="muted")
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.ylabel("Count")
        plt.xlabel(col)
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.show()
        
# Relationship with 'stroke'
def plot_binary_vs_stroke(data, binary_columns):
    for col in binary_columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=col, y='stroke', data=data, palette="muted")
        plt.title(f"Stroke Rate by {col.capitalize()}", fontsize=14)
        plt.ylabel("Stroke Rate")
        plt.xlabel(col.capitalize())
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.show()

# Columns to analyze
binary_columns = ['hypertension', 'heart_disease']

# Plot distributions and relationship with stroke
plot_binary_distributions(data, binary_columns)
plot_binary_vs_stroke(data, binary_columns)


# Insights on Binary Variables:
# 
# 1-Hypertension:
# Majority of patients do not have hypertension.
# Patients with hypertension show a higher stroke rate than those without.
# 
# 
# 2-Heart Disease:
# A small proportion of patients have heart disease.
# Those with heart disease have a notably higher stroke rate compared to those without.
# 
# 
# Both variables are significant risk factors for stroke, as indicated by the higher stroke rates among affected individuals.
Now that we’ve covered the binary variables and their relationships with stroke, the next step is to analyze how other features (numerical and categorical) relate to the target variable.
# In[54]:


#Let’s begin with the numerical features and their relationship to stroke.

# Visualizing relationships between numerical features and the target variable 'stroke'
def plot_numerical_vs_stroke(data, numerical_columns):
    for col in numerical_columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=data, x='stroke', y=col, palette="Set2")
        plt.title(f"{col.capitalize()} by Stroke Outcome", fontsize=14)
        plt.ylabel(col.capitalize())
        plt.xlabel("Stroke (0 = No, 1 = Yes)")
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.show()

# Numerical columns to analyze
numerical_columns = ['age', 'avg_glucose_level', 'bmi']

# Plot relationships
plot_numerical_vs_stroke(data, numerical_columns)


# Insights on Numerical Features and Stroke:
# 
# 1-Age:
# Patients who had a stroke are generally older compared to those who did not.
# The median age for stroke cases is significantly higher.
# 
# 2-Average Glucose Level:
# Stroke patients tend to have higher glucose levels on average.
# There is a wider range of glucose levels among stroke cases.
# 
# 3-BMI:
# No significant difference in the median BMI between stroke and non-stroke groups.
# However, outliers with high BMI are observed in both groups.

# In[55]:


# # analyzing the relationship between categorical features (e.g., gender, work_type, etc.) and stroke


# In[56]:


# Visualizing relationships between categorical features and the target variable 'stroke'
def plot_categorical_vs_stroke(data, categorical_columns):
    for col in categorical_columns:
        plt.figure(figsize=(8, 5))
        stroke_rates = data.groupby(col)['stroke'].mean().sort_values(ascending=False)
        sns.barplot(x=stroke_rates.index, y=stroke_rates.values, palette="coolwarm")
        plt.title(f"Stroke Rate by {col.capitalize()}", fontsize=14)
        plt.ylabel("Stroke Rate")
        plt.xlabel(col.capitalize())
        plt.xticks(rotation=45)
        plt.show()

# Categorical columns to analyze
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Plot relationships
plot_categorical_vs_stroke(data, categorical_columns)


# Insights on Categorical Features and Stroke:
# 
# 1-Gender:
# No significant difference in stroke rates between males and females.
# Very few patients are categorized as "Other," making it hard to draw conclusions for this group.
# 
# 2-Ever Married:
# Patients who are married have a noticeably higher stroke rate compared to those who are not married.
# 
# 3-Work Type:
# The highest stroke rate is observed in patients who have never worked.
# Self-employed individuals also show a slightly elevated stroke risk compared to those in private or government jobs.
# 
# 4-Residence Type:
# Stroke rates are fairly balanced between urban and rural residents.
# 
# 5-Smoking Status:
# Patients who smoke or formerly smoked exhibit higher stroke rates.
# The "Unknown" category still holds a large portion of the data and could introduce uncertainty.

# # II)  QUERIES
Below are some queries we'll adress so we can extract deeper insights from the data 

a)Hypertension and Stroke:
What percentage of stroke patients have hypertension?

b)Age Groups and Stroke:
How does the stroke rate vary across different age groups?

c)Smoking Status and Stroke:
Among smokers, how does the stroke rate compare to non-smokers and those with "Unknown" status?

d)Combined Health Conditions:
What percentage of stroke patients have both hypertension and heart disease?

e)Work Type and Stroke Risk:
Which work type has the highest stroke risk?

Let's start by addressing these queries one by one and sharing the results.

# In[57]:


# Query 1: Percentage of stroke patients with hypertension
hypertension_stroke_rate = data[data['stroke'] == 1]['hypertension'].mean() * 100

# Query 2: Stroke rate by age group
data['age_group'] = pd.cut(data['age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
stroke_by_age_group = data.groupby('age_group')['stroke'].mean() * 100

# Query 3: Stroke rate by smoking status
stroke_by_smoking_status = data.groupby('smoking_status')['stroke'].mean() * 100

# Query 4: Percentage of stroke patients with both hypertension and heart disease
stroke_combined_conditions = len(data[(data['stroke'] == 1) & (data['hypertension'] == 1) & (data['heart_disease'] == 1)]) / len(data[data['stroke'] == 1]) * 100

# Query 5: Stroke risk by work type
stroke_by_work_type = data.groupby('work_type')['stroke'].mean() * 100

# Display results
query_results = {
    "Hypertension Stroke Rate (%)": hypertension_stroke_rate,
    "Stroke Rate by Age Group (%)": stroke_by_age_group,
    "Stroke Rate by Smoking Status (%)": stroke_by_smoking_status,
    "Stroke Rate with Both Conditions (%)": stroke_combined_conditions,
    "Stroke Rate by Work Type (%)": stroke_by_work_type
}

query_results


# 1-Hypertension and Stroke:
# 26.51% of stroke patients have hypertension, highlighting its significant role as a risk factor.
# 
# 2-Age Groups and Stroke:
#    -Stroke rates increase with age:
#     0–20 years: 0.20%
#     21–40 years: 0.49%
#     41–60 years: 4.10%
#     61–80 years: 12.96%
#     81–100 years: 19.83%
#         
# 3-Smoking Status and Stroke:
# 
#     -Stroke rates by smoking status:
#         Unknown: 3.04%
#         Formerly smoked: 7.91%
#         Never smoked: 4.76%
#         Smokes: 5.32%
#             
# 4-Combined Health Conditions:
#     5.22% of stroke patients have both hypertension and heart disease, indicating the compounded risk of these conditions.
# 
# 5- Work Type and Stroke Risk:
#     Stroke rates by work type:
#         Government job: 5.02%
#         Never worked: 0%
#         Private: 5.09%
#         Self-employed: 7.94%
#         Children: 0.29%

# # III) Data Cleaning and Feature Engineering

# In[58]:


#first Step : 

#Analyzing the missing values in the bmi column.
#Choosing an appropriate imputation strategy (e.g., median, mean, or other methods) based on the distribution of bmi.
#Imputing the missing value


# In[59]:


# Visualizing the distribution of BMI (including missing values)
plt.figure(figsize=(8, 5))
sns.histplot(data['bmi'], kde=True, bins=30, color='blue')
plt.title("Distribution of BMI (including missing values)", fontsize=14)
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

# Analyzing missing values in 'bmi'
missing_bmi = data['bmi'].isnull().sum()
missing_bmi_percentage = (missing_bmi / len(data)) * 100

missing_bmi, missing_bmi_percentage


# Analysis of Missing Values in BMI:
# 
# Missing Count:
# 201 rows have missing BMI values.
# 
# Percentage:
# This accounts for approximately 3.93% of the dataset.
# 
# Distribution:
# The distribution of BMI (excluding missing values) appears approximately normal
Imputation Plan:
    
To handle the missing values, I propose using the median BMI value, as it's less sensitive to outliers than the mean and the distribution seems fairly symmetric.
# In[60]:


# Imputing missing values in 'bmi' with the median
bmi_median = data['bmi'].median()
data['bmi'].fillna(bmi_median, inplace=True)

# Verifying that there are no more missing values in 'bmi'
missing_bmi_after = data['bmi'].isnull().sum()
missing_bmi_after


# In[62]:


# Dropping the 'id' column as it is irrelevant for prediction
data.drop(columns=['id'], inplace=True)

# Confirming the column removal
data.columns.tolist()


# In[63]:


# Encoding categorical variables using one-hot encoding
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Confirming the new encoded columns
data_encoded.columns.tolist()


# In[64]:


data_encoded


# In[65]:


from sklearn.preprocessing import StandardScaler

# Columns to scale
numerical_features = ['age', 'avg_glucose_level', 'bmi']

# Initializing the scaler
scaler = StandardScaler()

# Scaling the numerical features
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

# Checking scaled values
data_encoded[numerical_features].describe()


# The numerical features (age, avg_glucose_level, bmi) have been successfully scaled using standardization,
# with a mean of 0 and a standard deviation of 1.
# This ensures that these features are on a comparable scale

# In[69]:


data_encoded
#it seems that we have to convert the categorical data into boolean (numerical)


# In[67]:


# Convert "True" and "False" to 1 and 0 if any exist in the dataset
data_encoded = data_encoded.replace({True: 1, False: 0})

# Verify if any non-numeric values remain in the dataset
non_numeric_columns = data_encoded.select_dtypes(include=['object']).columns.tolist()

# Display the non-numeric columns to ensure everything is numerical now
non_numeric_columns


# In[68]:


data_encoded


# # MODEL TRAINING

# In[70]:


from sklearn.model_selection import train_test_split

# Splitting the dataset into features (X) and target (y)
X = data_encoded.drop(columns=['stroke'])
y = data_encoded['stroke']

# Splitting into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Confirming the split
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Training Set:
# 4,088 samples, 20 features.
# 
# Test Set:
# 1,022 samples, 20 features.

# training multiple models (e.g., Logistic Regression, Random Forest, etc.) and evaluate their performance

# In[71]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Dictionary to store evaluation results
evaluation_results = {}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results
    evaluation_results[model_name] = {
        "Accuracy": accuracy,
        "Classification Report": report
    }

# Display evaluation results
evaluation_results


Model Training and Evaluation Results:
    
1-Logistic Regression:
    Accuracy: 95.11%
    Class Imbalance Issue: The model struggles to identify stroke cases (1), with a precision and recall of 0.0 for the minority class.

2-Random Forest:
    Accuracy: 95.11%
    Performs slightly better than Logistic Regression in identifying stroke cases, but the recall for class 1 (stroke) is only 2%. This indicates the model is still biased towards the majority class.

Key Observation:
Both models suffer from the class imbalance in the dataset, where non-stroke cases dominate. This leads to poor performance in identifying the minority class (stroke = 1).Next Steps:
To address the imbalance, I propose:

Resampling the Data:

Apply oversampling (e.g., SMOTE) or undersampling techniques to balance the classes.
Re-train the Models:

Train the models on the balanced dataset to improve their ability to detect stroke cases.I will proceed by doing a manual oversampling of the minority class (stroke =1) 
this will involve duplicating minority class samples to balance the dataset 
# # Balancing the Dataset

# In[75]:


# Separate minority and majority classes
minority_class = data_encoded[data_encoded['stroke'] == 1]
majority_class = data_encoded[data_encoded['stroke'] == 0]

# Oversample the minority class by duplicating its samples to match the majority class size
minority_oversampled = minority_class.sample(len(majority_class), replace=True, random_state=42)

# Combine the oversampled minority class with the majority class
balanced_data = pd.concat([majority_class, minority_oversampled])

# Shuffle the data to mix the classes
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and target for the balanced dataset
X_balanced = balanced_data.drop(columns=['stroke'])
y_balanced = balanced_data['stroke']

# Verify class distribution after manual oversampling
balanced_class_distribution = y_balanced.value_counts()

balanced_class_distribution


# The dataset is now balanced, with equal numbers of samples in both classes:
# 
# Class 0 (No Stroke): 4,861 samples.
# Class 1 (Stroke): 4,861 samples.

# In[76]:


# Re-train models on the balanced dataset
evaluation_results_balanced = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_balanced, y_balanced)
    
    # Make predictions on the original test set (X_test, y_test)
    y_pred_balanced = model.predict(X_test)
    
    # Evaluate the model
    accuracy_balanced = accuracy_score(y_test, y_pred_balanced)
    report_balanced = classification_report(y_test, y_pred_balanced, output_dict=True)
    
    # Store results
    evaluation_results_balanced[model_name] = {
        "Accuracy": accuracy_balanced,
        "Classification Report": report_balanced
    }

# Display evaluation results on the balanced dataset
evaluation_results_balanced


# Model Performance After Balancing the Dataset:
#     
# 1)Logistic Regression:
# Accuracy: 75.44%
# The model's recall for the minority class (stroke = 1) significantly improved to 84%, indicating it is much better at detecting stroke cases.
# Precision for the minority class remains low at 14.7%, which reflects some false positives.
# 
# 2)Random Forest:
# Accuracy: 100%
# Achieved perfect precision, recall, and F1-score for both classes. However, this may indicate overfitting, especially when tested on a small, imbalanced test set.

# # NEXT STEPS :
#     
# 1) Analyze Model Interpretability:
# Extract feature importances from the Random Forest model to understand which features contribute most to predictions.
# 
# 2)Evaluate Further on Unseen Data:
# Validate these results with cross-validation or an external validation set to check for overfitting.

# In[78]:


# Extracting feature importances from the Random Forest model
rf_model = models["Random Forest"]
feature_importances = pd.DataFrame({
    'Feature': X_balanced.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette="viridis")
plt.title("Feature Importances from Random Forest Model", fontsize=16)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Displaying the top feature importances
feature_importances.head(10)


# Feature Importance Analysis:
#     
# The Random Forest model identifies the following features as most important for predicting strokes:
# Age:
# The strongest predictor of stroke.
# 
# Average Glucose Level:
# A critical feature related to health conditions.
# 
# BMI (Body Mass Index):
# Significantly impacts stroke predictions.
# 
# Age Group (61–80):
# This specific age bracket contributes notably to predictions.
# 
# Age Group (21–40):
# Indicates that younger individuals still have some risk factors.
# 
# Ever Married (Yes):
# Marital status adds to the prediction, possibly as a proxy for lifestyle or healthcare access.
# 
# Hypertension:
# A well-known stroke risk factor.
# 
# Gender (Male):
# Slightly impacts predictions.
# 
# Smoking Status (Never Smoked):
# Indicates health behaviors influencing stroke likelihood.
# 
# Residence Type (Urban):
# Minimal but still relevant.

# Observations:
# The top three features (age, glucose levels, BMI) align with known medical risk factors for strokes, adding validity to the model's predictions.

# In[79]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation for both models on the balanced dataset
cv_results = {}

for model_name, model in models.items():
    # Cross-validation with 5 folds
    scores = cross_val_score(model, X_balanced, y_balanced, cv=5, scoring='accuracy')
    cv_results[model_name] = {
        "Mean Accuracy": scores.mean(),
        "Standard Deviation": scores.std()
    }

# Display cross-validation results
cv_results


# # Cross-Validation Results:
# 
# 1-Logistic Regression:
# Mean Accuracy: 79.38%
# Standard Deviation: 1.05%
# Performance is stable but slightly lower compared to the training-test evaluation.
# 
# 2-Random Forest:
# Mean Accuracy: 99.13%
# Standard Deviation: 0.10%
# Consistently high performance across folds, but the small standard deviation suggests potential overfitting.
# 
# Observations:
# Logistic Regression shows balanced generalization and could be a robust choice.
# Random Forest achieves very high accuracy, but further external validation may be needed to confirm it's not overfitting

# In[ ]:




