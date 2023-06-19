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

# Usage
Clone the repository: 
- git clone https://github.com/your-username/your-repo.git
- Navigate to the project directory: cd customer-churn-prediction
- Run the Jupyter Notebook
- Open the customer_churn_prediction.ipynb notebook and follow the step-by-step instructions to execute the code.
Note: Make sure to update the file paths in the notebook if your dataset is located in a different directory.

# Results
The final model achieved an accuracy of 0.7631%, precision of 0.6569%, recall of  0.6226%, and an F1 score of 0.6393% on the test set. These metrics indicate the performance of the model in predicting customer churn. Further details about the evaluation metrics and analysis can be found in the notebook.

Conclusion
Customer churn prediction is an important task for businesses, as it allows proactive measures to be taken to retain customers. This project demonstrates the application of machine learning techniques to predict churn in a telecommunications company. The trained model can be used to make churn predictions on new customer data and assist in customer retention efforts.

# Acknowledgments
This project was inspired by the Customer Churn Prediction dataset on Kaggle. We would like to thank the contributors of the dataset for providing valuable data for analysis.

# References
1. Kaggle
- Title: Telco Customer Churn
- Author: BLASTCHAR
- URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

2. YouTube Video
- Title: Exploratory Data Analysis in Pandas | Python Pandas Tutorials
- Author: Alex The Analyst
- URL: https://youtu.be/Liv6eeb1VfE

3. ChatGPT
ChatGPT is a large language model developed by OpenAI. It is powered by GPT-3.5, an advanced natural language processing model. ChatGPT is designed to generate human-like responses based on the provided input. This project utilizes ChatGPT for conversational interactions and text generation tasks. The model was trained by OpenAI, and its knowledge is current up to September 2021.
For more information about ChatGPT and its capabilities, please refer to the OpenAI documentation and guidelines.
