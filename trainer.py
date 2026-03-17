import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pickle

import gzip
import json


MAX_REVIEWS_PER_KATEGORIE = 10000     # Anzahl Reviews pro Kategorie die verarbeitet werden
MAX_FEATURES = 5000                   # Anzahl von Features für TF-IDF Vektor 





#Methodendefinitionen

def load_jsonl_gz_limited(file_path, max_samples):
    data = []
    r = 0
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
                r += 1
                if r >= max_samples:
                    break
            except:
                continue
    return pd.DataFrame(data)


def make_confusion_matrix(matrix, filename, fmt, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax if vmax is not None else matrix.max())
    labels = ["1", "2", "3", "4", "5"]
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(labels)

    for i in range(5):
        for j in range(5):
            ax.text(j, i, fmt(matrix[i, j]), ha="center", va="center", color="black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()










# Laden
# alle drei Files in DataFrame verpacken
print("Lade Daten...")
files = [
    "datasets/Amazon_Fashion.jsonl.gz",
    "datasets/Arts_Crafts_and_Sewing.jsonl.gz",
    "datasets/Electronics.jsonl.gz",
]

alle_df = []
for path in files:
    df_temp = load_jsonl_gz_limited(path, MAX_REVIEWS_PER_KATEGORIE)
    df_temp["category"] = path.split("/")[-1].replace(".jsonl.gz", "")
    alle_df.append(df_temp)

df = pd.concat(alle_df, ignore_index=True)











# Vorverarbeitung
# Konkatenieren - und alle Buchstaben klein - Ratingzahl zu integer
print("Vorverarbeitung läuft...")
df["cleaned_text"] = (df["title"] + " " + df["text"]).str.lower()
df["rating"] = df["rating"].astype(int)










# Undersampling
# Alle Rating-Klassen werden auf gleiche Anzahl Beispiele reduziert
lowest_count = df["rating"].value_counts().min()
undersampled_df = []

for rating in range(1, 6):
    rating_df = df[df["rating"] == rating]
    rating_df = rating_df.sample(n=lowest_count, random_state=42)
    undersampled_df.append(rating_df)

# Code sortiert Einträge nach Klassen, deshalb müssen wir wieder mischen
df_balanced = pd.concat(undersampled_df, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)












# Train/Test Split
# Daten vorbereiten (80/20 Split)
X = df_balanced["cleaned_text"]
y = df_balanced["rating"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)














# Training
# Eigentliches Training - Grenzen werden mit logistischer Regression gefittet.
print("Training...")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=2)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs"))
])

pipeline.fit(X_train, y_train)










# Evaluation
#Klassifikationsergebnise Kategorie und Gesamt
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
print("Accuracy:")
print(accuracy)









# Konfusionsmatrix
#Erzeugen der Ausgabe der Konfusuionsmatritzen im Stammordner
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

make_confusion_matrix(cm, "confusion_matrix.png", fmt=lambda v: str(v))
make_confusion_matrix(cm_normalized, "confusion_matrix_normalized.png", fmt=lambda v: f"{v:.1%}", vmax=1)

print("\nKonfusionsmatrizen wurden gespeichert als:")
print("  - confusion_matrix.png")
print("  - confusion_matrix_normalized.png")








# Speichern
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModell wurde gespeichert als model.pkl")