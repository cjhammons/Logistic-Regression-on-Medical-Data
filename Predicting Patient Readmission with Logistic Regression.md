# Research Question

We will build a model for predicting whether or not a patient will be readmitted to the hospital within 30 days of their release. We will look at patient demographics and health conditions to make this prediction.

# Methodology

Logistic Regression is the method of choice here because we are trying to predict a binary outcome: either "yes" or "no" (1 or 0). 

Logistic Regression relies on the following assumptions:
- The dependent variable is in fact binary
- All of the observations are independent
- There is no Multicollinearity among explanatory variables
- There are no extreme outlisers
- There is a Linear Relationship Between Explanatory Variables and the Logit of the Response Variable
- The sample size is sufficiently large

[Source](https://www.statology.org/assumptions-of-logistic-regression/)

## Tools

Python in Jupyterlab was used to write the code for this analysis. There are many statistics libraries written in python that make what we are planning to do very simple:
- ``Numpy`` and ``pandas`` for standard dataframe and numerical operations
- ``Sklearn`` and ``statsmodels`` for building the regression models
- ``matplotlib``, ``yellowbrick``, and ``sns`` for visualizations. 


```python
%%capture

# Install Packages
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install sklearn
!pip install statsmodels
!pip install yellowbrick
!pip install seaborn
```


```python
# Import Packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix




import random

random.seed(10)
```

# Data Preparation

We start with a clean dataset of 10,000 observations. We're going to do some additional cleaning to optimize it for our purposes.


```python
df = pd.read_csv("data/medical_clean.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CaseOrder</th>
      <th>Customer_id</th>
      <th>Interaction</th>
      <th>UID</th>
      <th>City</th>
      <th>State</th>
      <th>County</th>
      <th>Zip</th>
      <th>Lat</th>
      <th>Lng</th>
      <th>...</th>
      <th>TotalCharge</th>
      <th>Additional_charges</th>
      <th>Item1</th>
      <th>Item2</th>
      <th>Item3</th>
      <th>Item4</th>
      <th>Item5</th>
      <th>Item6</th>
      <th>Item7</th>
      <th>Item8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>C412403</td>
      <td>8cd49b13-f45a-4b47-a2bd-173ffa932c2f</td>
      <td>3a83ddb66e2ae73798bdf1d705dc0932</td>
      <td>Eva</td>
      <td>AL</td>
      <td>Morgan</td>
      <td>35621</td>
      <td>34.34960</td>
      <td>-86.72508</td>
      <td>...</td>
      <td>3726.702860</td>
      <td>17939.403420</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Z919181</td>
      <td>d2450b70-0337-4406-bdbb-bc1037f1734c</td>
      <td>176354c5eef714957d486009feabf195</td>
      <td>Marianna</td>
      <td>FL</td>
      <td>Jackson</td>
      <td>32446</td>
      <td>30.84513</td>
      <td>-85.22907</td>
      <td>...</td>
      <td>4193.190458</td>
      <td>17612.998120</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>F995323</td>
      <td>a2057123-abf5-4a2c-abad-8ffe33512562</td>
      <td>e19a0fa00aeda885b8a436757e889bc9</td>
      <td>Sioux Falls</td>
      <td>SD</td>
      <td>Minnehaha</td>
      <td>57110</td>
      <td>43.54321</td>
      <td>-96.63772</td>
      <td>...</td>
      <td>2434.234222</td>
      <td>17505.192460</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>A879973</td>
      <td>1dec528d-eb34-4079-adce-0d7a40e82205</td>
      <td>cd17d7b6d152cb6f23957346d11c3f07</td>
      <td>New Richland</td>
      <td>MN</td>
      <td>Waseca</td>
      <td>56072</td>
      <td>43.89744</td>
      <td>-93.51479</td>
      <td>...</td>
      <td>2127.830423</td>
      <td>12993.437350</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>C544523</td>
      <td>5885f56b-d6da-43a3-8760-83583af94266</td>
      <td>d2f0425877b10ed6bb381f3e2579424a</td>
      <td>West Point</td>
      <td>VA</td>
      <td>King William</td>
      <td>23181</td>
      <td>37.59894</td>
      <td>-76.88958</td>
      <td>...</td>
      <td>2113.073274</td>
      <td>3716.525786</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



## Drop Irrelavant Data

We begin the data preparation by removing irrelevant variables. We removed the following:
- CaseOrder
- Customer_id
- Interaction
- UID
- City
- State
- County
- Zip
- Lat
- Lng
- Interaction
- TimeZone
- Additional_charges
- Job
- Item1 
- Item2 
- Item3 
- Item4 
- Item5 
- Item6 
- Item7 
- Item8
- TotalCharge 


```python
df = df.drop(['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 
             'State', 'County', 'Zip', 'Lat', 'Lng', 'Interaction', 'TimeZone', 
              'Additional_charges', 'Job', 'Item1', 'Item2', 'Item3', 'Item4', 
             'Item5', 'Item6', 'Item7', 'Item8', 'TotalCharge'], axis=1)
```

## Make all data numeric

Logistic Regression requires numeric values to work, we will modify the data with dummy variables to accomodate this.


```python
readmis = df['ReAdmis']

df = pd.get_dummies(df)

# We want ReAdmis, our target variable, to be a single column
df = df.drop(['ReAdmis_No', 'ReAdmis_Yes'], axis=1)
df = df.assign(ReAdmis=readmis)
df['ReAdmis'] = pd.Categorical(df['ReAdmis'])
df['ReAdmis'] = df['ReAdmis'].cat.codes
```

## Splitting Data

We'll first divide the data into our target ``y``, ``ReAdmis``, and our predictors ``X``.

Feature | Type
--- | ---
Population | numeric
Area | categorical
Children| numeric
Age| numeric
Income| numeric
Marital| categorical
Gender| categorical
VitD_levels| numeric
Doc_visits| numeric
Full_meals_eaten| numeric
vitD_supp| numeric
Soft_drink| categorical
Initial_admin| categorical
HighBlood| categorical
Stroke| categorical
Complication_risk| categorical
Overweight| categorical
Arthritis| categorical
Diabetes| categorical
Hyperlipidemia| categorical
BackPain| categorical
Anxiety| categorical
Allergic_rhinitis| categorical
Reflux_esophagitis| categorical
Asthma| categorical
Services| categorical
Initial_days| numeric

All data labled "categorical" has of course been split into dummy variables.


```python
X = df.loc[:,df.columns!='ReAdmis']
y = df['ReAdmis']
```

We further split the data into training and testing data. We'll also apply a scaler.


```python
scaler = preprocessing.StandardScaler().fit(X)
scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled, y, test_size=0.2, random_state=42)
```

# Models

We'll start with an initial model using all of the predictors. Then we'll reduce the predictors and compare the reduced models to the initial one.

## Initial Model


```python
initial_model = LogisticRegression(random_state=0).fit(X_train, y_train)
initial_model.predict(X_test)
initial_score = initial_model.score(X_test, y_test)
```

## Data Reduction

We'll use Recursive Feature Elimination to reduce the data.


```python
selector = RFE(initial_model, n_features_to_select=10, step=3)
selector = selector.fit(X_train, y_train)

best_variables = []
sort = selector.ranking_.sort()

for i, val in enumerate(selector.ranking_):    
    if val == 1:
        best_variables.append(X.columns[i-1])

print("\n\nSelected reduced features:")
for b in best_variables:
    print('- %s' % b)
```

    
    
    Selected reduced features:
    - Services_MRI
    - Population
    - Children
    - Age
    - Income
    - VitD_levels
    - Doc_visits
    - Full_meals_eaten
    - vitD_supp
    - Initial_days


## Reduced Model

Now we rebuild the model with the reduces list of variables from the previous section.


```python
X_reduced = df[best_variables]

scaler = preprocessing.StandardScaler().fit(X_reduced)
scaled = scaler.transform(X_reduced)

X_train, X_test, y_train, y_test = train_test_split(scaled, y, test_size=0.2)
reduced_model = LogisticRegression(random_state=0).fit(X_train, y_train)
reduced_model.predict(X_test)
reduced_score = reduced_model.score(X_test, y_test)
```

Using only the five selected variables we were able to create a model that is only .003% worse than the full model, showing the impact of just these 5 variables.

# Results Summary


## Model Accuracy

We will look at the scores of the initial and reduced models and compare them.


```python
print('Initial model\'s score: %.3f, ' % initial_score)
print('Reduced model\'s score: %.3f, ' % reduced_score)
print('Difference: %.3f' % (reduced_score - initial_score))

print('\nCoefficients:')
for i, c in enumerate(reduced_model.coef_[0]):
    print('- %s: %.3f' % (X_reduced.columns[i], c))
```

    Initial model's score: 0.983, 
    Reduced model's score: 0.979, 
    Difference: -0.004
    
    Coefficients:
    - Services_MRI: 0.234
    - Population: 0.110
    - Children: 0.104
    - Age: 0.015
    - Income: -0.031
    - VitD_levels: 0.064
    - Doc_visits: 0.043
    - Full_meals_eaten: 0.053
    - vitD_supp: 0.001
    - Initial_days: 14.723


## Confusion Matrix
We will look at the confusion matrix to get a better idea of the model's accuracy.


```python
confusion_matrix(y_test, reduced_model.predict(X_test))
```




    array([[1248,   24],
           [  18,  710]])



Here is a more human-readable table:

| | Predicted Positive | Predicted Negative
--- | --- | ---
**Actual Positive** | 1209 | 22 
**Actual Negative** | 19 | 750

1959/2000 Of the test observations were true positives/negatives. Or 98%

## Statistical Significance

We were able to get within 4 thousandths of the accuracy of the full model with only the 10 variables we reduced to, an inconsequential change. With this information we now know that if a hospital wants to estimate the total cost of a visit they need only look at the reduced variables such as MRI services and Initial Days.

## Limitations

With only 10,000 patient records to work with there is a chance this model has been biased towards a particular conclusion. For example in another set of patients a different pre-existing condition such as Anxiety or a demographic attribute such as Gender may contribute more to ReAdmission. 

## Recommendations

The biggest contributer to readmission according to our model is the initial length of a patient's visit. Therefore to prevent readmissions hospitals should pay extra attention to patients that have already spent considerable time in the hospital.
