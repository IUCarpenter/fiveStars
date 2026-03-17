import pickle
import numpy as np

# Modell laden
#Laden der vortrainierten (mit Trainer.py) Modell/Pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Modell geladen. Strg+C zum beenden\n")

# CLI-Abfrage
while True:
    review = input("Gib eine Review zur Vorhersage ein: ").strip()
    
    if not review:
        continue

    prediction = model.predict([review])[0]
    probabilities = model.predict_proba([review])[0]
    confidence = np.max(probabilities)

    print(f"→ {prediction} Sterne  (P={confidence:.2f})\n")