#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[3]:


import pandas as pd
import numpy as np


# In[6]:


# Open dataset
df = pd.read_csv("vgsales.csv")


# In[7]:


# Display first few rows of dataset
df.head()


# In[8]:


# Show info on each column
df.info()


# In[9]:


# Gives us number of columns and rows in dataset
df.shape


# In[10]:


# Calculates summary descriptive stats for our dataset
df.describe()


# # Question 2

# In[13]:


# We want to first check for any missing values in our dataset
df.isnull().sum()


# In[19]:


# We will remove the year rows as it won't affect our analysis
df.dropna(subset = ["Year"])
df


# In[22]:


#Now we want to replace missing publisher values with mode

mode_publisher = df['Publisher'].mode()[0]
df_cleaned_categorical = df.fillna({'Publisher': mode_publisher})

print("Number of missing values in Publisher after cleaning:", df_cleaned_categorical['Publisher'].isnull().sum())


# In[16]:


# We want to now get rid of any duplicate values in our dataset
df.drop_duplicates()
df


# # Question 3

# In[2]:


pip install seaborn


# In[6]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[19]:


# We want to first visualize distribution of all numeric predictor attributes
df = pd.read_csv("vgsales.csv")

# Select numeric predictor attributes
numeric_predictors = df.select_dtypes(include='number')

# Now we will see the distrbution using boxplots as they help us identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=numeric_predictors)
plt.title('Distribution of Numeric Predictor Attributes')
plt.xlabel('Attributes')
plt.ylabel('Values')
plt.xticks(rotation=45)  # Rotating the x-axis allows for better readability
plt.show()


# In[12]:


# Now we want to visualize the distribution of numeric target attribute
sns.histplot(df['NA_Sales'], kde=True)
plt.title('Distribution of NA_Sales')
plt.xlabel('NA Sales')
plt.ylabel('Frequency')
plt.show()


# In[ ]:





# In[15]:





# # Question 4

# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# for categorical data
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("vgsales.csv")

# boxplot to visualise numerical against categorical attribute

plt.figure(figsize=(12, 6))
sns.barplot(x='Platform', y='Global_Sales', data =df) 
plt.title('Global Sales by Platform')
plt.xlabel('Platform')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=45)
plt.show()


# # QUestion 5

# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("vgsales.csv")

# selecting only numerical attributes 
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))

# creatint heatmap and modifying the annotation, colour, float and width of the chart
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numeric Attributes')
plt.show()


# # Question 6

# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("vgsales.csv")

# We want to plot genre and global sales
plt.figure(figsize=(12, 6))
sns.barplot(x='Genre', y='Global_Sales', data=df) 
plt.title('Average Global Sales by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Global Sales (in millions)')
plt.xticks(rotation=45)
plt.show()


# # Question 7

# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# this is a seperate dataset used only for this question
df = pd.read_csv("vgsales_2.csv")

# we need to first name our bin ranges and labels
bins = [0, 1, 10, df['Global_Sales'].max()]
labels = ['Low Sales', 'Medium Sales', 'High Sales']

# Create a new categorical target attribute based on the numeric target attribute
df['Global_Sales_Category'] = pd.cut(df['Global_Sales'], bins=bins, labels=labels, right=False)

# Visualize the distribution of another numeric attribute for each category of the transformed attribute
plt.figure(figsize=(10, 6))
sns.boxplot(x='Global_Sales_Category', y='Year', data=df) 
plt.title('Distribution of Year by Global Sales Category')
plt.xlabel('Global Sales Category')
plt.ylabel('Year')
plt.xticks(rotation=45)
plt.show()


# # Question 8

# In[33]:


pip install scikit-learn


# In[39]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("vgsales.csv")

# Initializing the LabelEncoder
label_encoder = LabelEncoder()

# we need to add the LabelEncoder to each categorical column
df['Platform'] = label_encoder.fit_transform(df['Platform'])
df['Genre'] = label_encoder.fit_transform(df['Genre'])
df['Publisher'] = label_encoder.fit_transform(df['Publisher'])
df['Name'] = label_encoder.fit_transform(df['Name'])  # Assuming 'Name' is a categorical column

print(df.head())





# # Question 9

# In[42]:


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("vgsales.csv")

# creating variable for my student ID
student_id = 22434792

# now we want to set the last non-zero digit
last_num = int(str(student_id)[-1])

# creating the ratio based on last non-zero digit on student ID
if last_num in [4, 6]:
    train_ratio = 0.6
    test_ratio = 0.4
elif last_num in [1, 9]:
    train_ratio = 0.9
    test_ratio = 0.1
else:
    train_ratio = 0.5
    test_ratio = 0.5

# random seed for matching whole student number without the x
random_seed = int(str(student_id).replace('x', ''))

# Outcome
train_set, test_set = train_test_split(df, train_size=train_ratio, test_size=test_ratio, random_state=random_seed)

print("Train ratio:", train_ratio)
print("Test ratio:", test_ratio)


# # Question 10

# In[46]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv("vgsales.csv")

# We need to first separate the numeric attributes
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# next step is to initiliaze the min-max scaler
scaler = MinMaxScaler()

# transform both train and test sets
train_set_scaled = scaler.fit_transform(train_set[numeric_columns.columns])
test_set_scaled = scaler.transform(test_set[numeric_columns.columns])

# print scaled train and test set
print("Scaled Train Set:")
print(train_set_scaled_df.head())
print("\nScaled Test Set:")
print(test_set_scaled_df.head())



# In[48]:


pip install statsmodels


# # Question 11

# In[49]:


import statsmodels.api as sm
from sklearn.feature_selection import f_regression
import numpy as np

df = pd.read_csv("vgsales.csv")

# Firstly, we need to utilize OLS regression using all our predictors

# our predictors
X = df.drop(columns=['Global_Sales']) 

# our target
y = df['Global_Sales']  

# we are adding a constant value to X for interception
X = sm.add_constant(X)

# now we want to fit the OLS model accordingly
ols_model = sm.OLS(y, X).fit()

# feature selection method based on modulo operation of my student ID
student_id = 22434792
feature_selection_method = student_id % 3

# feature selection method
# best subset
if feature_selection_method == 0:  
    pass
# forward stepwise
elif feature_selection_method == 1:  
    pass
# backwards stepwise
elif feature_selection_method == 2:  
    pass

# print the OLS model
print(ols_model.summary())


# # Question 12

# In[54]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("vgsales.csv")

# predictors (X) and target variable (y)
X = df.drop(columns=['Global_Sales'])
y = df['Global_Sales']

# train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression model
ridge_model = Ridge(alpha=1.0)

# training the Ridge Regression model
ridge_model.fit(X_train, y_train)

# predictions on test set
ridge_pred = ridge_model.predict(X_test)

# calculate R-square
ridge_r2 = r2_score(y_test, ridge_pred)
print("Ridge Regression R-squared score:", ridge_r2)

# random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# training the Random Forest Regression model
rf_model.fit(X_train, y_train)

# predictions on the test set
rf_pred = rf_model.predict(X_test)

# Calculate R-square
rf_r2 = r2_score(y_test, rf_pred)
print(rf_r2)

# final step is to plot the results
plt.plot(range(1, 11), ridge_scores, label='Ridge Regression')
plt.plot(range(10, 101, 10), rf_scores, label='Random Forest Regression')
plt.xlabel('Model Complexity')
plt.ylabel('Cross-validation R-squared')
plt.title('Cross-validation Performance')
plt.legend()
plt.show()



# In[ ]:




