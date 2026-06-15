import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(r"C:\Users\sahuj\Downloads\Credit_Card_Dataset.csv")
categorical_cols = ['Gender', 'Marital_Status', 'Education_Level', 'Employment_Status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
numerical_cols = ['Age', 'Annual_Income', 'Credit_Score', 'Total_Transactions_Last_Year', 'Total_Spend_Last_Year']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
print(df_encoded.head())
from sklearn.model_selection import train_test_split
y = df_encoded['Defaulted']
X = df_encoded.drop(columns=['Defaulted', 'Customer_ID'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("Training the Random Forest model ")
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("\n--- Model Evaluation ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
print("Applying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Original training defaults: {sum(y_train == 1)}")
print(f"SMOTE training defaults: {sum(y_train_smote == 1)}")
print("\nTraining the Random Forest model on SMOTE data...")
rf_model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_model_smote.predict(X_test)
print("\n--- NEW Model Evaluation (With SMOTE) ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_smote))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote))
probabilities = rf_model_smote.predict_proba(X)
default_probabilities = probabilities[:, 1]
df['Risk_Score'] = (default_probabilities * 100).round(2)
df['Risk_Flag'] = rf_model_smote.predict(X)
print(df[['Customer_ID', 'Risk_Score', 'Risk_Flag']].head(10))
df.to_csv(r"C:\Users\sahuj\Downloads\FinSight_Scored_Portfolio.csv", index=False)