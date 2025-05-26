Welcome! This project is all about preparing the Titanic dataset for machine learning and analysis. I cleaned the data, handled missing values, converted categorical features into numbers, scaled the data, and removed outliers to make it ready for modeling.

What I Did
1.Loaded and explored the data
Checked the structure, looked for missing values, and took a peek at summary stats using pandas 
python
import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')
print(df.info())
print(df.describe())

2.Handled missing values
Filled missing Age values with the median age.
Filled missing  cabin and embarked values with the mode as it is character based 
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


3.Replaced missing Cabin info with 'Unknown'.
Converted categories to numbers
Pulled the first letter from the Cabin field and turned it into one-hot columns.
One-hot encoded Sex and Embarked columns.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


4.Scaled the features
Used standard scaling on Age and Fare so they’re on the same level.

5.Found and removed outliers
Made boxplots to see outliers, then removed them using the IQR method.

Files Included
1.Titanic-Dataset.csv — The original dataset
2.Task1.ipynb — Jupyter notebook with all my preprocessing steps
3.screenshots/ — Pictures like boxplots and outputs
4.README.md — This file you’re reading now

 Libraries Needed
Make sure you have these installed:
1.pandas
2.numpy
3.matplotlib
4.seaborn
5.scikit-learn
