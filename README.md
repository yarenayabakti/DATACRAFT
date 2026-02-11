# DATACRAFT
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# CSV YOLU

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "train.csv")

df = pd.read_csv(file_path)


# TEMEL BİLGİLER

print("\nSatır / Kolon:", df.shape)
print("\nLoan_Status Dağılımı:")
print(df['Loan_Status'].value_counts())

print("\nEksik Veri:")
print(df.isnull().sum())


# GEREKSİZ KOLON

df = df.drop(columns=["Loan_ID"])


# DEPENDENTS DÜZELTME

df["Dependents"] = df["Dependents"].replace("3+", "3")


# SAYISAL KOLONLAR

numeric_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Dependents"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())


# KATEGORİK KOLONLAR

categorical_cols = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# ENCODING

le = LabelEncoder()

for col in df.select_dtypes(include=["object", "string"]):
    df[col] = le.fit_transform(df[col])


# SON NaN KONTROL

print("\nKalan NaN Sayısı:")
print(df.isnull().sum())


# MODEL

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# TAHMİN

pred = model.predict(X_test)

print("\nİlk 5 Tahmin:")
for i in range(5):
    print("Gerçek:", y_test.iloc[i], "Tahmin:", pred[i])
