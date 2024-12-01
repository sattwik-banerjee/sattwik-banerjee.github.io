# Final Project : Detecting Heart Disease with Random Forrest and Neural Network

Sattwik Banerjee

Department of Atmospheric and Oceanic Sciences, UCLA

AOS C111: Introduction to Machine Learning for the Physical Sciences

Dr. Alexander Lozinski

December 9, 2024


# Introduction 

Over the past few decades, poor health has risen to be widely acknowledged as one of the main causes of death worldwide and especially in the United States of America. Of the many potential chronic illnesses and diseases, Heart Disease is the leading cause. The usual regimen to fight heart disease is by altering a person’s lifestyle and adding dietary restrictions as often people diagnosed with heart disease have very high levels of cholesterol and blood pressure from a life of unhealthy habits (Jones & Greene, 2013). However the average age of a person diagnosed with heart disease is around 65% for men and 72% for women, with the odds only increasing as time passes (Etudo, 2024). A big question that has been thrown around in the medical field is how can people susceptible to heart disease take preventative care when the average diagnosis age is around ⅔ a person’s life expectancy?

![](assets/cardiovascular-disease-deaths-by-age-2.png){: width="750" }
*Figure 1: Number of Cardiovascular Deaths by Age Globally.[1]*

One proposed solution to this problem has been deemed to be the implementation of Machine Learning. The field of Machine Learning has been taking the world by storm with the continuous advance in applications of Machine Learning in all aspects of life. As Machine Learning continues to expand its reach, many health care experts believe that with Machine Learning, healthcare can be improved for all, especially in regards to Heart Disease (Baht & Gupta, 2023). In this paper I will explore the application of Machine Learning within the medical field to detect heart disease within a patient.

# Data

The dataset used in this analysis is the Heart Disease Dataset, sourced from the UCI MachineLearning Repository. This dataset contains records from individuals undergoing clinical testing for heart disease. It includes 13 features related to patient demographics, medical history, and clinical measurements, alongside a target variable indicating the presence or absence of heart disease.
The goal is to predict whether an individual has heart disease based on these features, using machine learning techniques.

The input variables present in this dataset are age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression, slope of the peak exercise ST segment, number of major vessels, and thalassemia. During the modeling process not all these features will be used to create the models, however they are all a part of the process to prepare the data.

Preparing the Data : To be able to create the best model and prediction possible, the data must first go through a “preprocessing” stage. In this stage the data is cleaned, meaning that the dataset is thoroughly checked for any and all values that would reduce the accuracy of the Machine Learning Model. For this particular dataset, my preprocessing step began with checking for missing or empty values in the dataset using the command 
```python
X.isna().sum()
```
where X represents the data frame containing the input features. After that command, if there appears to be any NaN values in the set, I then use the command 
```python
X.dropna()
```
to remove the rows that contain those values. 

Next, I make sure that all the features were scaled appropriately to improve the models performance using the code below : 
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Finally, in this dataset the target variable has values from 0-4 where 0 symbolizes no heart disease, and 1-4 represents heart disease within a patient. The values 1-4 serve to display the severity in heart disease, but for the purpose of simply detecting heart disease I will save all values greater than or equal to 1 as 1.

After performing these steps the preprocessing stage has been completed and data is ready to be used to create the models.

![](assets/Uknown.png){: width="700" }

*Figure 1: Input and Target Data from the UCI Dataset [2].*

# Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

# Results

Figure X shows... [description of Figure X].

# Discussion

From Figure X, one can see that... [interpretation of Figure X].

# Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

# References
[1] DALL-E 3

[back](./)

