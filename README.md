# Customer Churn Prediction
This project aims to predict customer churn in a telecommunications company. Customer churn refers to the phenomenon where customers switch to a different service provider or stop using a service altogether. By predicting churn, the company can take proactive measures to retain customers and reduce revenue loss.

# Project Overview
The project involves the following steps:
- Data Understanding: The dataset used for this project contains information about customers, such as demographic details, service usage, and churn status. Exploratory data analysis (EDA) was performed to gain insights into the data and understand its characteristics.

- Data Preprocessing: The data required preprocessing steps, including handling missing values, encoding categorical variables, and scaling numerical features. The processed data was then split into training and testing sets.

- Model Selection and Training: Several machine learning models were evaluated for their performance in predicting churn. Logistic Regression and Random Forest were selected as the final models for this project. The models were trained on the training set and optimized using appropriate evaluation metrics.

- Model Evaluation: The trained models were evaluated on the testing set using various evaluation metrics, including accuracy, precision, recall, and F1 score. Additional evaluation metrics such as ROC AUC and average precision were also considered. The model with the best performance was selected for deployment.

# Requirements
The project was implemented using Python 3. The following libraries were used:
- pandas
- numpy
- scikit-learn
# To install the required libraries, run the following command:
pip install -r requirements.txt
