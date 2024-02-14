import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import os

load_dotenv()

data_path = os.environ.get('POWERLIFTING_DATA_PATH')

# Load the dataset with specified dtype for columns with DtypeWarning
dtype_dict = {'Dots': str, 'Wilks': str, 'Glossbrenner': str, 'Goodlift': str}
data = pd.read_csv(data_path, dtype=dtype_dict)

# Select relevant columns
selected_columns = ['Best3DeadliftKg', 'Best3SquatKg', 'Best3BenchKg', 'Sex']
df = data[selected_columns]

# Drop rows with missing values in the selected columns
df = df.dropna(subset=['Best3DeadliftKg', 'Best3SquatKg', 'Best3BenchKg', 'Sex'])

# Convert 'Sex' column to numerical labels (assuming 'M' for Male and 'F' for Female)
df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})

# Check for NaN values in the target variable
if df['Sex'].isnull().any():
    df = df.dropna(subset=['Sex'])

# User interaction: Prompt the user for input
user_deadlift = float(input("Enter the weight for the Best Deadlift (in Kg): "))
user_squat = float(input("Enter the weight for the Best Squat (in Kg): "))
user_bench = float(input("Enter the weight for the Best Bench (in Kg): "))

# Create a new DataFrame with user input
user_data = pd.DataFrame({'Best3DeadliftKg': [user_deadlift],
                          'Best3SquatKg': [user_squat],
                          'Best3BenchKg': [user_bench]})

# Split the data into features (X) and target (y)
X = df[['Best3DeadliftKg', 'Best3SquatKg', 'Best3BenchKg']]
y = df['Sex']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the user input
user_pred = clf.predict(user_data)

# Display the predicted gender
if user_pred[0] == 0:
    print("The model predicts Male.")
else:
    print("The model predicts Female.")
