# Customer-Churn-Prediction-Using-ML

# Introduction

Our project is on predicting Customer Churn. Customer churn refers to identifying customers who are likely to stop doing business with a company. Predicting churn allows businesses to take proactive measures to retain those customers, potentially saving a significant amount of revenue. Churn can negatively impact a company's brand image and reputation. Thus, identifying churn risk factors allows companies to better target their marketing and retention efforts.

# Dataset Description

Dataset Source: Telco Customer Churn (Kaggle)
Our dataset consists of 24 features and the description of each feature is given below -
-customerID: This refers to the unique identifier for each customer.
-gender: This indicates the customer's gender, either male or female. 
-SeniorCitizen: This indicates whether the customer is a senior citizen (65 years old or older), with 1 representing yes and 0 representing no. 
-Partner: This indicates whether the customer has a partner, with "Yes" or "No" as the options.
-Have Dog : Whether the customer owns a dog or not.
-Dependents: This indicates whether the customer has dependents, with "Yes" or "No" as the options. 
-tenure: This represents the number of months the customer has been subscribed to the company's services.
-Watch Netflix: Represents whether the customer watches Netflix or not
-PhoneService: This indicates whether the customer has phone service, with "Yes" or "No" as the options.
-MultipleLines: This indicates whether the customer has multiple phone lines, with "Yes", "No", or "No phone service" as the options. 
-InternetService: This indicates the customer's internet service provider, with "DSL", "Fiber optic", or "No" as the options. 
-OnlineSecurity: This indicates whether the customer has online security services, with "Yes", "No", or "No Internet service" as the options. 
-Use Tiktok: Indicates whether the customer uses Tiktok or not
-OnlineBackup: This indicates whether the customer has online backup services, with "Yes", "No", or "No Internet service" as the options. 
-DeviceProtection: This indicates whether the customer has device protection services, with "Yes", "No", or "No Internet service" as the options.
-TechSupport: This indicates whether the customer has technical support services, with "Yes", "No", or "No Internet service" as the options.
-FeelsLonely: Whether the customer feels lonely
-StreamingTV: This indicates whether the customer has streaming TV services, with "Yes", "No", or "No Internet service" as the options. 
-StreamingMovies: This indicates whether the customer has streaming movie services, with "Yes", "No", or "No Internet service" as the options.
-Contract: This indicates the type of contract the customer has with the company, which could be "Month-to-month", "One year", or "Two years".
-Paperless billing: This indicates whether the customer has opted for paperless billing, with "Yes" or "No" as the options.
-PaymentMethod: This indicates the customer's preferred payment method, which could be "Electronic check", "Mailed check", "Bank transfer (automatic)", or "Credit card (automatic)". 
-Monthly charges: This refers to the monthly subscription cost that the customer currently pays.
-TotalCharges: This represents the total amount the customer has paid to the company so far.

## Target Variable
And we have a target variable or classification column:
Churn: This indicates whether the customer has churned (terminated) their service with the company, with "Yes" or "No" as the options.
Here among all the features, other than three numerical features(Tenure, Monthly charges, Total charges) all other features are categorical.
Customer churn prediction falls into a classification problem
This is because The target variable in customer churn prediction is typically binary: either a customer churns (yes/1) or they do not (no/0). This binary nature of the target variable makes it ideal for classification algorithms designed to predict discrete categories.
Regression algorithms, on the other hand, are typically used to predict continuous target variables, such as customer lifetime value or the amount of money a customer will spend.
Here in customer churn prediction, the goal is to identify customers who are most likely to churn so that interventions can be implemented to prevent them from leaving. This requires a model that can effectively distinguish between churned and non-churned customers, which is a core function of classification algorithms.
Regression algorithms, while capable of predicting values, are not as well-suited for identifying specific groups within data.



# Data-Preprocessing
## Dropping Unnecessary Columns or Features
In our dataset, the customer ID feature doesn't seem to contribute anything as all the customers have unique IDs, so we will not find any pattern here. As a result, we dropped the column using .drop().
Along with that, the features HaveDog, WatchNetflix, UseTiktok, and FeelsLonely don’t have any values, all the entries of these columns are null. So these columns were dropped as well.
Dropping rows with Duplicate values
Then, the entries which had duplicate values were dropped using .drop_duplicated(inplace=True). It was necessary to drop them because duplicate entries can introduce bias into the dataset, causing the model to overfit to those specific entries and generalize poorly to unseen data. This can lead to inaccurate predictions and reduced model performance.
## Removing rows with Null values
In some of the entries, there were some null values, or for some entries, there were some features without any value. So we removed those entries or rows without removing a whole column for it. It was done using .dropna(axis = 0, subset = ['gender', 'Partner', 'Dependents', 'tenure', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']).
## Imputing row entries that contain Null values
However, we did not remove rows with null values in the StreamingTV or TotalCharges column, as it would result in a great data loss. Instead, we imputed those null fields. For the StreamingTV feature, we used  SimpleImputer(missing_values=np.nan, strategy='most_frequent').In this Imputation technique missing values in a dataset are replaced with the most frequent value observed for each feature.
Then, for the TotalCharges column, the Imputation function used was - SimpleImputer(missing_values=np.nan, strategy='mean') This replaces missing values with the mean of the column.
## Encoding Categorical Features
In addition to that, the three features of our dataset - Tenure, Monthly Charges, TotalCharges consisted of numerical values and the rest are Categorical features. So, these categorical values were encoded using the Ordinal Encoder. This was necessary because these machine learning algorithms typically work with numerical data, and categorical data is stored as text or labels. Encoding converts the categorical values into numerical representations that the algorithms can understand and process.
Lastly, our target being a categorical feature was also encoded using the LabelEncoder due to the reasons mentioned above.
Ordinal encoding converts categorical data with an inherent order into numerical values. Unlike nominal encoding, which treats all categories as equally distinct and assigns them unique numerical values regardless of any inherent order, ordinal encoding preserves the relative importance and order of categories.
Label Encoder analyzes the categorical data and creates a unique integer identifier for each distinct category it encounters.
## Feature Scaling
The three features of our dataset - Tenure, Monthly Charges, TotalCharges consisted of numerical values. So, these numerical values were scaled using the MinimaxScaler function which scales each feature to a range between 0 and 1.
Feature scaling was necessary as many machine learning algorithms rely on calculating distances between data points. Features with large ranges can dominate these calculations, making it difficult for the algorithm to find the optimal solution. Hence, scaling ensures that all features have a similar range, providing a level playing field for the algorithm and improving its ability to converge to the best solution
## Dataset Splitting
We split our processes dataset into a 70:30 ratio, using 70% of the data to train the model and the rest 30% of the data to test the model.
The training data is used to build and refine the model. The model learns from the patterns and relationships in the training data, adjusting its internal parameters to make accurate predictions.
The testing data is used to assess the ability of the trained model to generalize to unseen data. This provides an unbiased estimate of the model's performance in real-world scenarios.

# Model Training and Testing
The four models used for the prediction are - Decision Tree, Random Forest, Logistic Regression, and Gaussian Naive Bayes.
## Decision Tree
A decision tree is a supervised machine-learning technique that can create a flowchart-like structure to make predictions based on the values of the input features. It can be used for both classification and regression problems, and it is easy to understand and interpret. A decision tree consists of nodes, branches, and leaves, where each node represents a test on a feature, each branch represents an outcome of the test, and each leaf represents a class label or a numerical value. A decision tree is built by recursively splitting the data into subsets based on the best feature and the best-split point until a stopping criterion is met. The best feature and the best-split point are determined by using a metric such as entropy, Gini index, or variance, which measures the impurity or the variation in the data. The goal is to find the feature and the split point that maximizes the information gain or minimizes the impurity or the variation in the data.
## Random Forest
Random forest is a supervised machine learning algorithm that combines multiple decision trees to create a “forest” of predictions. It can handle both classification and regression problems, and it is robust to noise and overfitting. Random forest works by taking random samples of data and features, and building a decision tree for each sample. Then, it aggregates the predictions of all the trees, either by taking the average (for regression) or the majority vote (for classification). Random forest is useful for dealing with complex and high-dimensional data, and it can also provide estimates of feature importance and error rates.
## Logistic Regression
Logistic regression is a supervised machine-learning technique that can predict the probability of a binary outcome based on one or more input features. It can be used for classification problems, such as spam detection, credit risk, or disease diagnosis. Logistic regression uses a logistic function, also known as a sigmoid function, to map the linear combination of the input features to a value between 0 and 1, which represents the probability of the positive class. Logistic regression can handle both categorical and numerical features, and it can also provide estimates of the significance and the effect of each feature on the outcome.
## Gaussian Naive Bayes
Gaussian naive bayes is a variant of naive bayes that follows Gaussian normal distribution and supports continuous data. It is a supervised machine learning technique that can predict the probability of a binary outcome based on one or more input features. It assumes that each feature is independent of the others and has a normal distribution with a mean and a standard deviation. It uses the Bayes theorem to calculate the posterior probability of each class given the features, and assigns the class with the highest probability.

# Comparison Analysis Between the  4 Models Used 

From the charts generated in the notebook, we can see that Model accuracy using Gaussian Naive Bayes is  74.51 %,79.25 % using Logistic Regression,  78.34 % using Random Forest, and 72.64 % using Decision Tree.
Thus, we can conclude that logistic regression performs better than the other models we have used.

# Conclusion
In conclusion, our project focused on predicting customer churn, a crucial aspect for businesses to proactively retain customers and safeguard revenue. We utilized the Telco Customer Churn dataset from Kaggle, comprising 24 features, and framed the prediction task as a binary classification problem.
The data preprocessing phase involved dropping unnecessary columns, handling duplicate values, and addressing missing data through imputation. Categorical features were encoded using the Ordinal Encoder, and the target variable (Churn) was encoded using LabelEncoder. Feature scaling was applied to ensure numerical features had a consistent range.
The dataset was split into a 70:30 ratio for training and testing, enabling model evaluation on unseen data. Four models—Decision Tree, Random Forest, Logistic Regression, and Gaussian Naive Bayes—were employed for prediction. Logistic Regression emerged as the most effective model, surpassing the others in accuracy.
Visualizations further illustrated the distribution of churn and showcased the comparative analysis of model performance. The findings suggest that logistic regression outperforms the other models in predicting customer churn, providing valuable insights for businesses to implement targeted interventions and retention strategies.







