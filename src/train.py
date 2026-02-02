import pandas as pd

# Load dataset
df = pd.read_excel("weather.xlsx")

print("DATA PREVIEW:")
print(df.head())

print("\nDATA INFO:")
print(df.info())

# Bersihkan nama kolom
df.columns = (
    df.columns
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("°", "")
    .str.replace("%", "percent")
    .str.replace("-", "_")
    .str.replace("/", "")
    .str.replace("(", "")
    .str.replace(")", "")
)


print("\nCLEANED COLUMNS:")
print(df.columns)

# Target & Distribusi Kelas
y = df["rainyn"]
X = df.drop("rainyn", axis=1)

# Hapus fitur yang berpotensi leakage
X = X.drop(columns=["rainmm"])

print("\nTarget distribution:")
print(y.value_counts())
print("\nTarget ratio:")
print(y.value_counts(normalize=True))



# Seleksi Kolom Missing Berat
missing_ratio = X.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.6].index

print("\nDropped columns (missing > 60%):")
print(cols_to_drop)

X = X.drop(columns=cols_to_drop)



# Train/Test Split (Anti Data Leakage)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Pisahkan Tipe Data
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

print("\nNumerical columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))


# Pipeline Preprocessing Profesional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


# Model Baseline
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Mesin ML Lengkap
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

# Training Model
print("\nTraining model...")
clf.fit(X_train, y_train)

# Evaluasi Model
from sklearn.metrics import classification_report, f1_score

y_pred = clf.predict(X_test)

print("\nF1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cek Feature Importance dan Data Leakage
leak_cols = ["rainmm"]
X = X.drop(columns=[c for c in leak_cols if c in X.columns])


import pandas as pd

feature_names = clf.named_steps["preprocess"].get_feature_names_out()
importances = clf.named_steps["model"].feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print(feat_imp.head(10))

print("\nColumns after leakage drop:")
print(X.columns)

