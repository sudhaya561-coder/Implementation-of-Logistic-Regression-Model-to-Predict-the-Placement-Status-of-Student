# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import Required Libraries

Step 2: Load the Dataset

Step 3: Copy Data & Drop Unwanted Columns

Step 4: Check Data Quality

Step 5: Encode Categorical Variables

Step 6: Define Features (X) and Target (y)

Step 7: Split into Training and Testing Sets

Step 8: Build and Train Logistic Regression Model

Step 9: Make Predictions

Step 10: Evaluate the Model

Step 11: Predict for a New Student

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: UDHAYA.S
RegisterNumber: 212225230287 
*/
```
~~~ python
# 1. Import Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 2. Load the Dataset
# NOTE: Change the file path if your CSV is in a different location
data = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")

# View first 5 rows
print("First 5 rows of the dataset:")
print(data.head())



# 3. Create a Copy and Drop Unwanted Columns
data1 = data.copy()

# Dropping 'sl_no' (serial number) and 'salary' (not needed for predicting placement)
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())



# 4. Check for Missing and Duplicate Values
print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())




# 5. Encode Categorical Variables using LabelEncoder
# Columns that are categorical (object type)
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())





# 6. Define Features (X) and Target (y)
# X = all columns except 'status'
X = data1.iloc[:, :-1]
# y = 'status' column
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())



# 7. Split the Dataset into Training and Testing Sets
# test_size=0.2 → 20% test data, 80% training data
# random_state=0 → same split every time (for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)



# 8. Create and Train the Logistic Regression Model
# solver='liblinear' works well for small datasets
lr = LogisticRegression(solver="liblinear")

# Train the model
lr.fit(X_train, y_train)



# 9. Make Predictions on the Test Set
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)




# 10. Evaluate Model Performance
# Accuracy: percentage of correctly predicted labels
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Classification Report: precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))





# 11. Predict Placement for a New Student
# Order of features must match X columns:
# ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s',
#  'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']

# Example student data (after encoding categorical values manually)
# NOTE: These categorical numeric codes depend on how LabelEncoder encoded them.
# Here we assume:
# gender: 1 (e.g., Male)
# ssc_p: 80
# ssc_b: 1
# hsc_p: 90
# hsc_b: 1
# hsc_s: 1
# degree_p: 90
# degree_t: 1
# workex: 0 (No work experience)
# etest_p: 85
# specialisation: 1
# mba_p: 85

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])

~~~

## Output:

<img width="1168" height="381" alt="Screenshot 2026-08-17 113907" src="https://github.com/user-attachments/assets/147fb406-0719-4fc2-b337-be7192cd0ee1" />
<img width="1038" height="376" alt="Screenshot 2026-08-17 113843" src="https://github.com/user-attachments/assets/73570d42-d497-470a-9055-0a8e747aef23" />
<img width="1233" height="361" alt="Screenshot 2026-08-17 113827" src="https://github.com/user-attachments/assets/2dd38780-38ba-46fd-8d39-b84c54ad9cad" />
<img width="1241" height="369" alt="Screenshot 2026-08-17 113808" src="https://github.com/user-attachments/assets/bd0663c1-2a08-4907-9337-5845af1b1980" />
<img width="1242" height="360" alt="Screenshot 2026-08-17 113745" src="https://github.com/user-attachments/assets/eca670aa-7077-49b7-969f-68b97eb929cd" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
