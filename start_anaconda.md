# Start in Anaconda

## 1) Environment anlegen
```bash
conda env create -f environment.yml
```

## 2) Environment aktivieren
```bash
conda activate scenario-tool
```

## 3) App starten
Wechsle in den Ordner mit `scenario_management_tool.py` und `scenario_data.json`.

```bash
streamlit run scenario_management_tool.py
```

## Hinweis
Die App nutzt jetzt den Dateipfad relativ zur Python-Datei selbst. Dadurch findet sie `scenario_data.json` auch dann korrekt, wenn du sie aus einer Anaconda-Umgebung oder aus einem anderen Arbeitsverzeichnis startest.
