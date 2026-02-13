import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# --- 1. DATA PREPARATION ---
dataset_path = "final_test.csv"
df_original = pd.read_csv(dataset_path)

def inject_data(n=25000, is_pants=False):
    rows = []
    # BMI logic for sizes
    size_map = {"S": (15, 18.5), "M": (18.5, 24), "L": (24, 27), "XL": (27, 31), "XXL": (31, 36), "XXXL": (36, 45)}
    
    for _ in range(n):
        h = np.random.uniform(150, 195)
        age = np.random.uniform(18, 65)
        h_m = h / 100
        
        for s_label, (low_b, high_b) in size_map.items():
            w = np.random.uniform(low_b * (h_m**2), high_b * (h_m**2))
            bmi = w / (h_m**2)
            
            if is_pants:
                # W/L Logic: Waist follows BMI, Length follows Height
                waist = int(round(bmi * 1.1 + 2))
                length = int(round(h * 0.18))
                rows.append([w, age, h, bmi, f"W{waist}/L{length}"])
            else:
                rows.append([w, age, h, bmi, s_label])
    return pd.DataFrame(rows, columns=['weight', 'age', 'height', 'bmi', 'size'])

# --- 2. TRAIN SHIRT MODELS ---
print("👕 Training Shirt Models...")
df_shirts = inject_data(is_pants=False)
le_shirt = LabelEncoder()
df_shirts['size_enc'] = le_shirt.fit_transform(df_shirts['size'])

scaler_s = StandardScaler()
X_s = df_shirts[['weight', 'age', 'height', 'bmi']]
y_s = df_shirts['size_enc']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.15)
X_train_s = scaler_s.fit_transform(X_train_s)

models = {
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=3),
    "decision_tree": DecisionTreeClassifier(max_depth=15),
    "logistic_regression": LogisticRegression(max_iter=2000),
    "svm": LinearSVC(dual=False)
}

stats = {}
for name, m in models.items():
    m.fit(X_train_s, y_train_s)
    joblib.dump(m, f'{name}_model.pkl')
    y_pred = m.predict(scaler_s.transform(X_test_s))
    stats[name] = {"Accuracy": accuracy_score(y_test_s, y_pred), "Report": classification_report(y_test_s, y_pred, output_dict=True)}

joblib.dump(scaler_s, 'scaler.pkl')
joblib.dump(le_shirt, 'label_encoder.pkl')
joblib.dump(stats, 'model_stats.pkl')

# --- 3. TRAIN PANTS MODEL ---
print("👖 Training Pants Model...")
df_pants = inject_data(is_pants=True)
le_pants = LabelEncoder()
df_pants['size_enc'] = le_pants.fit_transform(df_pants['size'])

scaler_p = StandardScaler()
X_p = df_pants[['weight', 'age', 'height', 'bmi']]
y_p = df_pants['size_enc']

m_pants = RandomForestClassifier(n_estimators=200, max_depth=20)
m_pants.fit(scaler_p.fit_transform(X_p), y_p)

joblib.dump(m_pants, 'pants_model.pkl')
joblib.dump(scaler_p, 'pants_scaler.pkl')
joblib.dump(le_pants, 'pants_le.pkl')
print("✅ All Models Saved Successfully!")