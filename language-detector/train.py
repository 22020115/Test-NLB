"""Buoc 3: huan luyen mo hinh nhan dien ngon ngu bang TF-IDF (character n-gram)."""
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

try:
    data = pd.read_csv("data/sentences.csv").dropna()
except FileNotFoundError:
    sys.exit("Khong thay data/sentences.csv - hay chay prepare_data.py truoc.")

# Chia 80% train / 20% test, giu ti le ngon ngu deu nhau (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    data["text"].str.lower(),
    data["lang"],
    test_size=0.2,
    random_state=42,
    stratify=data["lang"],
)
print(f"Train: {len(X_train)} cau | Test: {len(X_test)} cau\n")


def make_vectorizer():
    # analyzer="char": cat theo ky tu (khong theo tu) - rat hop de nhan dien ngon ngu
    return TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)


models = {
    "Naive Bayes": make_pipeline(make_vectorizer(), MultinomialNB(alpha=0.01)),
    "Logistic Regression": make_pipeline(
        make_vectorizer(), LogisticRegression(max_iter=1000)
    ),
}

best_name, best_model, best_acc = None, None, -1.0
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"=== {name}: accuracy = {acc:.4f} ===")
    if acc > best_acc:
        best_name, best_model, best_acc, best_pred = name, model, acc, pred

print(f"\nMo hinh tot nhat: {best_name} ({best_acc:.4f})\n")
print(classification_report(y_test, best_pred, zero_division=0))

labels = sorted(data["lang"].unique())
cm = pd.DataFrame(
    confusion_matrix(y_test, best_pred, labels=labels),
    index=[f"that:{l}" for l in labels],
    columns=[f"doan:{l}" for l in labels],
)
print("Confusion matrix (hang = dap an that, cot = mo hinh doan):")
print(cm)

joblib.dump(best_model, "model.joblib")
print("\nDa luu mo hinh vao model.joblib")

# Thu vai cau tu go
for s in ["Xin chào, tôi đang tìm việc", "Hello, how are you?", "Bonjour tout le monde"]:
    print(f"{s!r} -> {best_model.predict([s.lower()])[0]}")