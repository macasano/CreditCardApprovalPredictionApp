#!/usr/bin/env python
# coding: utf-8

# ## C964 Capstone

# In[3]:


# Standard inputs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
#%matplotlib inline

# In[4]:

# Load dataset from GitHub repo
url = "https://raw.githubusercontent.com/macasano/CreditCardApprovalPredictionApp/refs/heads/main/clean_dataset.csv"
df = pd.read_csv(url)
df.head()

# In[5]:


# Split into X/y and drop column "CreditScore"
X = df.drop(columns=["Approved", "CreditScore"], errors="ignore")
y = df["Approved"]

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
#X_train.shape, X_test.shape, y_train.shape, y_test.shape

# In[6]:


# Values must be converted to numerical values
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define categorical features
cf = ["Industry", 
      "Ethnicity", 
      "Citizen"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot, 
                                  cf)], remainder="passthrough")
transformed_X = transformer.fit_transform(X)
#transformed_X

# In[7]:


# Refit model
np.random.seed(45)
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.20)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# In[8]:


# Accuracy
model.score(X_test, y_test)

# In[9]:


with open("../c964_capstone/model.pkl", "wb") as f:
    pickle.dump(model, f)

# In[10]:


# Evaluation
from sklearn.ensemble import RandomForestClassifier

np.random.seed(52)

X = df.drop(columns=["Approved", "CreditScore"], errors="ignore")
y = df["Approved"]

X = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# In[11]:


model.score(X_test, y_test)

# In[12]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(52)

X = df.drop(columns=["Approved", "CreditScore"], errors="ignore")
y = df["Approved"]

X = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[13]:


model.score(X_test, y_test)

# In[14]:


cross_val_score(model, X, y)

# In[15]:

# Evaluations
np.random.seed(43)

single = model.score(X_test, y_test)

cross_val = np.mean(cross_val_score(model, X, y))

#single, cross_val

# In[16]:


#Confusion matrix w/ seaborn heatmap

# In[17]:


from sklearn.metrics import confusion_matrix

y_predict = model.predict(X_test)

confusion_matrix(y_test, y_predict)

# In[18]:


pd.crosstab(y_test, y_predict, rownames=["actual"], colnames=["predicted"])

# In[19]:


#import sys
#!conda install --yes --prefix {sys

# In[20]:


# Generate confusion matrix
import seaborn as sns
conf_mat = confusion_matrix(y_test, y_predict)

sns.set(font_scale=1)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Approved', 'Approved'], yticklabels=['Not Approved', 'Approved'])

plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
plt.show()

# In[21]:


orgData = pd.read_csv(url)

# In[22]:


# Correlation Matrix
from sklearn.preprocessing import LabelEncoder

X_copy = orgData.copy()
label_cols = ["Industry", "Ethnicity", "Citizen"]

for col in label_cols:
    if col in X_copy.columns:
        le = LabelEncoder()
        X_copy[col] = le.fit_transform(X_copy[col])
    else:
        print(f"Column '{col}' not found in DataFrame.")

correlation_matrix = X_copy.corr()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()

# In[35]:


# Histogram for each feature
numeric_cols = X_copy.select_dtypes(include=['int64', 'float64']).columns

num_cols = len(numeric_cols)
n_cols = 3 
n_rows = (num_cols + n_cols - 1) // n_cols

plt.figure(figsize=(n_cols * 5, n_rows * 4))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(X_copy[col], bins=30, color='blue', edgecolor='black')
    plt.title(col)
    #plt.xlabel(col)
    #plt.ylabel("Frequency")

plt.tight_layout()

plt.show()

# In[23]:


from sklearn.preprocessing import LabelEncoder

X_copy = orgData.copy()
label_cols = ["Industry", "Ethnicity", "Citizen"]

for col in label_cols:
    if col in X_copy.columns:
        le = LabelEncoder()
        X_copy[col] = le.fit_transform(X_copy[col])
    else:
        print(f"Column '{col}' not found in DataFrame.")

correlation_matrix = X_copy.corr()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[26]:

# User Interface
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

with open("../c964_capstone/model.pkl", "rb") as f:
    model = pickle.load(f)

# Transform data and fit to model
df = pd.read_csv(url)
features = df.drop(columns=["Approved", "CreditScore"]).columns.tolist()

categorical_cols = ["Industry", "Ethnicity", "Citizen"]

transformer = ColumnTransformer(
    transformers=[("one_hot", OneHotEncoder(), categorical_cols)],
    remainder="passthrough"
)
transformer.fit(df[features])

st.title("Mock Credit Card Application")

# Input application fields
gender = st.selectbox("What is your gender?", ["male", "female"])
age = st.number_input("How old are you?", min_value=18, max_value=120, value=25)
debt = st.number_input("How much debt do you have?", min_value=0.0, value=0.0, format="%.2f")
married = st.selectbox("Are you married?", ["yes", "no"])
bank_customer = st.selectbox("Are you a bank customer?", ["yes", "no"])
industry = st.selectbox("What industry do you work in?", [
    "Industrials", "Materials", "CommunicationServices", "Transport", "InformationTechnology",
    "Financials", "Energy", "Real Estate", "Utilities", "ConsumerDiscretionary", "Education",
    "ConsumerStaples", "Healthcare", "Research"])
ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Latino", "Other"])
years_employed = st.number_input("How many years have you been employed?", min_value=0.0, value=0.0, format="%.1f")
prior_default = st.selectbox("Have you defaulted on a payment?", ["yes", "no"])
employed = st.selectbox("Are you employed?", ["yes", "no"])
drivers_license = st.selectbox("Do you have a driver's license?", ["yes", "no"])
citizen = st.selectbox("Citizenship", ["ByBirth", "ByOtherMeans", "Temporary"])
zipcode = st.text_input("What is your zipcode? (5 digits)", max_chars=5)
income = st.number_input("What is your monthly income?", min_value=0.0, value=1000.0, format="%.2f")

# Converts yes and no to binary
def to_binary(x):
    return 1 if x in ["yes", "female"] else 0

# Submit button
if st.button("Submit Application"):
    if len(zipcode) != 5 or not zipcode.isdigit():
        st.error("ZipCode must be 5 digits.")
    elif age < years_employed:
        st.error("Age cannot be less than years employed.")
    else:
        input_data = {
            "Gender": [to_binary(gender)],
            "Age": [age],
            "Debt": [debt],
            "Married": [to_binary(married)],
            "BankCustomer": [to_binary(bank_customer)],
            "Industry": [industry],
            "Ethnicity": [ethnicity],
            "YearsEmployed": [years_employed],
            "PriorDefault": [to_binary(prior_default)],
            "Employed": [to_binary(employed)],
            "DriversLicense": [to_binary(drivers_license)],
            "Citizen": [citizen],
            "ZipCode": [int(zipcode)],
            "Income": [income]
        }

        df_input = pd.DataFrame(input_data)
        X_transformed = transformer.transform(df_input[features])

        # Make a prediction
        prediction = model.predict(X_transformed)
        result = "Approved" if prediction[0] == 1 else "Denied"
        st.success(f"Prediction: {result}")


