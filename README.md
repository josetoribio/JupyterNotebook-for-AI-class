# 🌸 Iris SVM Classifier & Visualization

This project demonstrates how to use **Support Vector Machines (SVM)** for **Iris flower classification** and visualize decision boundaries using **Seaborn** and **Matplotlib**.  
It focuses on the relationship between **petal width** and **petal length** to distinguish between the three Iris species.

---

## 📊 Overview

The **Iris dataset** is a classic dataset used in statistics and machine learning.  
In this project, we will:

- Visualize the distribution of Iris flower species.
- Train a **Linear SVM** classifier using Scikit-learn.
- Plot the **decision boundary** and **margins** of the separating hyperplane.
- Test predictions for different flower measurements.

---

## 🧠 Technologies Used

- Python 3  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## ⚙️ Step 1 — Load and Visualize the Data

```python
%matplotlib inline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('IRIS.csv')

# Visualize petal dimensions by species
sns.lmplot('petal_width', 'petal_length',
           data=data,
           hue='species',
           palette='Set1',
           fit_reg=False,
           scatter_kws={"s": 50})

🧩 Step 2 — Train an SVM Classifier
from sklearn import svm

# Select features and labels
points = data[['petal_width', 'petal_length']].values
result = data['species']

# Train Linear SVM
clf = svm.SVC(kernel='linear')
clf.fit(points, result)

🔍 Step 3 — Examine Model Parameters
print("Vector of weights (w):", clf.coef_[0])
print("Intercept (b):", clf.intercept_[0])
print("Indices of support vectors:", clf.support_)
print("Support vectors:\n", clf.support_vectors_)
print("Number of support vectors for each class:", clf.n_support_)


Example Output:

Vector of weights (w) = [-0.7 -1.1]
b = 3.2799
Number of support vectors for each class = [1 13 12]

🧮 Step 4 — Plot the Decision Boundary and Margins
import numpy as np

# Extract model parameters
w = clf.coef_[0]
b = clf.intercept_[0]
slope = -w[0] / w[1]

# Define the hyperplane
xx = np.linspace(0, 4)
yy = slope * xx - (b / w[1])

# Define margins
s1 = clf.support_vectors_[0]
s2 = clf.support_vectors_[-1]
yy_down = slope * xx + (s1[1] - slope * s1[0])
yy_up = slope * xx + (s2[1] - slope * s2[0])

# Plot results
sns.lmplot('petal_width', 'petal_length', data=data, hue='species', palette='Set1',
           fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='green', label='Decision Boundary')
plt.plot(xx, yy_down, 'k--', label='Margin')
plt.plot(xx, yy_up, 'k--')
plt.legend()
plt.title("SVM Decision Boundary for Iris Dataset")
plt.show()

🔮 Step 5 — Make Predictions
print(clf.predict([[3, 3]])[0])
print(clf.predict([[4, 0]])[0])
print(clf.predict([[2, 2]])[0])
print(clf.predict([[1, 2]])[0])


Output:

Iris-versicolor
Iris-setosa
Iris-versicolor
Iris-setosa

📈 Results Summary
Input (petal_width, petal_length)	Predicted Species
(3, 3)	Iris-versicolor
(4, 0)	Iris-setosa
(2, 2)	Iris-versicolor
(1, 2)	Iris-setosa
✅ Key Takeaways

Linear SVM separates Iris species effectively based on petal dimensions.

The decision boundary clearly divides species classes.

Support vectors define the optimal margin for classification.

💡 Future Improvements

Add model accuracy and confusion matrix evaluation.

Experiment with non-linear kernels (RBF, polynomial).

Extend visualization to 3D plots using all four Iris features.

Compare SVM with other classifiers (KNN, Decision Tree, Logistic Regression).

📚 References

Scikit-learn SVM Documentation

Iris Dataset UCI Repository
plt.title("Petal Width vs Petal Length by Species")
plt.show()
