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

![](assets/it_plots.png){: width="800" }

*Figure 1: Input and Target Data from the UCI Dataset [2].*

# Modelling

I employed two distinct machine learning models to classify the data in my project: a *random forest* and an *artificial neural network* (ANN).
A random forest is an ensemble learning method that combines the predictions of several decision trees to improve overall performance. Given that my data set was labeled, this constitutes a supervised learning problem. Specifically, I focused on a binary classification task where the model predicts either 0 (indicating 'no heart disease') or 1 (indicating ‘heart 'disease'). Using scikit-learn’s pre-existing library `RandomForestClassifier`, I implemented the Random Forest parameterized with `n_estimators=150`, `max_depth=15`, `min_samples_leaf=2`, `class_weight='balanced'`. 
For the ANN I utilized the `Tensorflow` library of python, specifically the keras interface. In the construction of the ANN, I chose an architecture of two hidden layers and one dropout layer to prevent overfitting configured with ReLU activations 
```python
model = Sequential([
   Dense(32, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
   Dense(64, activation='relu'),  # Hidden layer
   Dropout(0.4),  # Dropout layer to prevent overfitting
   Dense(1, activation='sigmoid')  # Output layer for binary classification
])
```


# Results

### Random Forest
After creating and running the Random Forest Model it had an accuracy of 88.3%. Here below are the Confusion Matrix for the model and additionally there is the plot of the most impactful input features on the model.
Interestingly based on the Feature Importance plot, we see that the two most impactful features in determining whether a patient has heart disease or not are thalach: maximum heart rate achieved and ca: number of major vessels (0-3) colored by fluoroscopy. 
This indicates that the biggest warning signs that a patient might have cardiovascular disease can be determined mainly by a person’s maximum heart rate achieved and number of major vessels (0-3) colored by fluoroscopy along with any combination of the following features. 
Below are the ROC (Receiver Operating Characteristic) curve and the Precision-Recall Curve. These plots serve to better visualize the performance of the model.

On the left is the plot of the ROC Curve, and on the right you see the Precision-Recall Curve. 
These plots help to visualize the performance of a model because of the metrics associated with each plot. For the ROC Curve we have the `AUC` which represents the area under the curve, and for the Precision-Recall Curve we have the `Average Precision` / `(AP)`. Both these metrics are scored on a scale from 0 - 1, the closer to 1 the better especially for Binary Classification tasks such as this one. We see that the `AUC = 0.95` and the `AP = 0.93` which means that the model performed quite well. 

### ANN
For the ANN, after creating and testing the model, it had an accuracy of 88%. Below is the Confusion Matrix for the model. 
When comparing the Confusion Matrices for both models we see that they are indeed quite similar. In both matrices, the false positives/negatives are relatively equal and the correctly predicted negatives and positives are similar as well, which I find to be quite interesting. 
Now below are the ROC and Precision-Recall Curves for the ANN model.
Once again the plots follow the same format, the left being the ROC and the right being the Precision-Recall. In comparison to the ROC and Precision-Recall for the Random Forest, the ANN performs slightly worse but not by all that much. The `AUC = 0.94` and the `AP = 0.91`. 


# Discussion

From Figure X, one can see that... [interpretation of Figure X].

# Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

# References
[1] Etudo, M. (2024, July 22). Heart attack age: Risk by age group. Medical News Today. https://www.medicalnewstoday.com/articles/heart-attack-age-range#:~:text=Statistics%20from%20the%20American%20Heart,attacks%20can%20happen%20to%20anyone

[2] Jones, D. S., & Greene, J. A. (2013, July). The decline and rise of coronary heart disease: Understanding public health catastrophism. American Journal of Public Health. U.S. National Library of Medicine. https://pmc.ncbi.nlm.nih.gov/articles/PMC3682614/

[3] Bhatt, C. M., Bansal, S., & Gupta, P. (2023, February 6). Effective heart disease prediction using machine learning techniques. MDPI. https://www.mdpi.com/1999-4893/16/2/88#:~:text=Using%20machine%20learning%20to%20classify,fatality%20caused%20by%20cardiovascular%20diseases

[4] Our World in Data. (n.d.). Number of deaths from cardiovascular diseases by age. Our World in Data. https://ourworldindata.org/grapher/cardiovascular-disease-deaths-by-age. Accessed November 30, 2024.

[back](./)

