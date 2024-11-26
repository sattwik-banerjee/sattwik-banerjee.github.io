# Final Project : Detecting Heart Disease with Random Forrest and Neural Network

Sattwik Banerjee

Department of Atmospheric and Oceanic Sciences, UCLA

AOS C111: Introduction to Machine Learning for the Physical Sciences

Dr. Alexander Lozinski

December 9, 2024


# Introduction 

Over the past few decades, poor health has risen to be widely acknowledged as one of the main causes of death worldwide and especially in the United States of America. Of the many potential chronic illnesses and diseases 

# Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

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

