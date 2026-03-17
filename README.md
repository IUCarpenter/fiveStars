Dieses Projekt trainiert ein NLP Modell in Python. Die Aufgabenstellung stammt aus einem IU-Projekt (Projekt: NLP).





Ziel: Mithilfe von Trainigsdaten aus dem Amazon Reviews 2023 Datensatzes wird ein Modell und eine Pipeline erzeugt.

Diese Pipeline ermöglicht die Klassifikation von Reviews in numerische Kennzahlen (1 bis 5 Sterne)



Das finale Modell liegt als model.pkl Datei mit in diesem Repository und kann verwendet werden.



Diese README-Datei dient als Installationsanleitung







Voraussetzungen:

Python 3.10+ muss auf dem Windows System installiert sein!!

Projektdateien müssen lokal im Projektverzeichnis liegen

model.pkl muss im Projektverzeichnis liegen





\----Optional: Für das trainieren eines neuen Modells: ------

Die Amazon Datensätze wurden heruntergeladen und liegen in

"datasets/"



Link:

https://amazon-reviews-2023.github.io/



Alternativ kann auch durch ausführen der "Trainingsdaten Herunterladen.bat" Datei der Download gestartet werden.

Das kann je nach Bandbreite lange dauern (mehr als 7 GB Daten)

\----------------------------------------------------





2\. ######### Installation und Anwendung #########



\- CLI im Projektverzeichnis öffnen.



\- Venv erzeugen:

"python -m venv nlpenv"



\- Venv aktivieren:

"nlpEnv\\scripts\\activate"



\- Requirements installieren:

"pip install -r requirements.txt"



\--------------------------------

Zum Trainieren:

\- Trainingsprogramm ausführen:

"python trainer.py"

\--------------------------------



Zum Vorhersagen:

\- Vorhersageprogramm ausführen:

"python runner.py"


