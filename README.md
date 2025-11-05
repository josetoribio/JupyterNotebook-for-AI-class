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
plt.title("Petal Width vs Petal Length by Species")
plt.show()
