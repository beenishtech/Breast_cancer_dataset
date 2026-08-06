import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# 1. Dataset read karein
df = pd.read_csv("breast_cancer_dataset.csv")

# 2. Features aur Target split
X = df.drop(columns=['target'])
y = df['target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pipeline (Scaling + Random Forest Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 4. Model train aur save karein
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'cancer_model.pkl')

print("Success: Model trained and saved as 'cancer_model.pkl'")