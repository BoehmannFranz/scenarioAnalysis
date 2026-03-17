# README – Szenario-Management Tool (Version 9)

## Projektüberblick

Dieses Tool ist ein interaktiver Prototyp für **Szenario-Management und strategische Verlaufssimulation**.  
Es dient dazu, Szenarien, Risiken und Gegenmaßnahmen entlang einer Zeitachse zu erfassen und deren Wirkung auf mehrere strategische Verläufe sichtbar zu machen.

Die aktuelle Arbeitsbasis ist **Version 9** des Prototyps.

Der Fokus des Tools liegt auf drei Fragen:

1. Welche **Szenarien** erzeugen Druck auf die strategische Entwicklung?
2. Wie stark entfernen sich strategische Verläufe von der **Nulllinie / Zeitachse**?
3. Welche **Gegenmaßnahmen** ziehen die strategischen Verläufe wieder zurück?

---

## Fachliche Grundidee

Im Zentrum des Modells steht eine **Zeitachse**, die sich an definierten **Phasen** orientiert.

Jeder strategische Verlauf beginnt zunächst bei **0** beziehungsweise an der **Zeitachse / Nulllinie**.

Danach wirken zwei Arten von Einflüssen:

### 1. Szenarien
Szenarien stellen Ereignisse oder Entwicklungen dar, die strategischen Druck erzeugen.  
Je nach **Intensität**, **Risikobewertung** und Zuordnung zu einem oder mehreren strategischen Verläufen entfernen sich diese Verläufe von der Zeitachse.

### 2. Gegenmaßnahmen
Gegenmaßnahmen wirken stabilisierend.  
Sie reduzieren die Auslenkung oder ziehen die strategischen Verläufe wieder zurück in Richtung Zeitachse.

---

## Visualisierungslogik

Version 9 enthält drei zentrale Darstellungen:

### 1. 3D-Körper
Der strategische Körper visualisiert mehrere Verläufe räumlich.  
Die Zeitachse bildet die Längslogik der Darstellung.  
Die einzelnen Verläufe können visuell hervorgehoben werden.  
Zoom und Rotation werden über das Interaktionsmenü gesteuert.

### 2. Ausgerollte Fläche
Diese Ansicht zeigt die Verläufe entlang der Zeit in einer abgewickelten Form.  
Hier lässt sich nachvollziehen, wie sich die einzelnen strategischen Verläufe entwickeln.  
Zusätzlich kann eine **Gesamtwirkung** eingeblendet werden.

### 3. Querschnitt
Der Querschnitt zeigt die Form des strategischen Körpers zu einem gewählten Zeitpunkt.  
Dadurch wird sichtbar, wie stark die Verläufe in diesem Moment voneinander abweichen.

---

## Zentrale Elemente des Tools

### Szenario
Ein Szenario besteht aus:
- Titel
- Phase
- Zeitpunkt / Position
- Beschreibung
- Einflussstärke / Intensität
- Risikostärke
- direktem Strategieeffekt
- Zuordnung zu einem oder mehreren strategischen Verläufen
- optionalen Kennzahlen

### Gegenmaßnahme
Eine Gegenmaßnahme besteht aus:
- Titel
- Phase
- Zeitpunkt / Position
- Beschreibung
- Wirksamkeit
- Strategiebeitrag
- Status
- Zuordnung zu einem oder mehreren strategischen Verläufen
- optionalen Kennzahlen

### Phase
Phasen strukturieren die Zeitachse in sinnvoll interpretierbare Abschnitte.  
Jede Phase hat:
- Name
- Start
- Ende
- Beschreibung

### Strategische Verläufe
Strategische Verläufe sind die Achsen bzw. Ecken des strategischen Körpers.  
In Version 9 sind mehrere Verläufe vordefiniert und können Szenarien oder Maßnahmen zugeordnet werden.

---

## Bedienung des Tools

### Linke Seite
Die linke Seite dient der Eingabe und Steuerung:
- Szenario-Steuerung
- Phasen definieren
- Szenarien erfassen
- Gegenmaßnahmen erfassen
- Speichern in JSON

### Rechte Seite
Die rechte Seite dient der Analyse:
- Metriken
- 3D-Körper
- ausgerollte Fläche
- Querschnitt
- Tabellenansicht für Szenarien und Gegenmaßnahmen

---

## Datenhaltung

Die Daten werden in einer Datei namens:

`scenario_data.json`

gespeichert.

Die Datei wird beim Speichern **überschrieben** und enthält den aktuellen Stand des Modells.

---

## Projektdatei Version 9

Die aktuelle Arbeitsdatei ist:

`scenario_management_tool_prism_v9.py`

Diese Version enthält:
- 3D-Körper
- Zoom- und Rotations-Slider
- ausgerollte Fläche
- Querschnitt
- Hervorhebung strategischer Verläufe
- Speicherung in JSON

---

## Installation

### Voraussetzungen
- Python in einer Anaconda- oder Conda-Umgebung
- Streamlit
- Plotly
- Pandas
- NumPy

### Beispiel mit Conda-Umgebung `dojo`
```bash
conda activate dojo
cd C:\Users\User\Desktop\Szenario
streamlit run scenario_management_tool_prism_v9.py
```

---

## Empfohlene Python-Pakete

```bash
pip install streamlit plotly pandas numpy
```

oder mit Conda:
```bash
conda install streamlit pandas numpy
pip install plotly
```

---

## Typischer Arbeitsablauf

1. Szenario auswählen oder neues Szenario anlegen  
2. Phasen auf der Zeitachse definieren  
3. Szenarien erfassen und strategischen Verläufen zuordnen  
4. Gegenmaßnahmen erfassen und Verläufen zuordnen  
5. Wirkung in 3D, ausgerollter Fläche und Querschnitt prüfen  
6. Modell in `scenario_data.json` speichern

---

## Zielbild des Tools

Das Tool ist als Entscheidungs- und Analysehilfe gedacht.  
Es soll nicht nur einzelne Ereignisse visualisieren, sondern die Frage unterstützen:

**Wie entwickelt sich strategischer Druck über die Zeit, und welche Maßnahmen stabilisieren das System wieder?**

Daraus ergeben sich mögliche Anwendungen in:
- Szenarioanalyse
- Risikobetrachtung
- Strategiemonitoring
- Krisenvorbereitung
- Managementkommunikation
- experimenteller Visualisierung komplexer Wirkzusammenhänge

---

## Aktueller Status des Prototyps

Version 9 ist ein visueller und funktionaler Prototyp.  
Sie ist geeignet für:
- konzeptionelle Erprobung
- Diskussion im Team
- visuelle Exploration
- erste Modellierung von Szenarien und Maßnahmen

Sie ist noch **kein fertiges Produktivsystem**.  
Insbesondere sind folgende Punkte perspektivisch weiter ausbaubar:
- Bearbeiten/Löschen einzelner Einträge
- feinere mathematische Wirklogik
- Rollen- und Rechtemodell
- Versionierung von Szenarien
- Vergleich mehrerer Szenario-Sets
- Export in Präsentationsformate
- persistente Datenbank statt JSON-Datei

---

## Bekannte Grenzen

- Plotly-3D-Interaktion ist funktional, aber in der Achsensteuerung nicht vollständig frei einschränkbar
- die Wirklogik ist prototypisch und nicht wissenschaftlich kalibriert
- JSON als Speicherformat ist einfach, aber nicht optimal für komplexe Mehrbenutzer-Szenarien
- große Mengen an Szenarien und Maßnahmen können die Lesbarkeit der 3D-Darstellung reduzieren

---

## Kurzbeschreibung in einem Satz

Dieses Tool visualisiert, wie Szenarien strategische Verläufe über die Zeit auslenken und wie Gegenmaßnahmen diese Verläufe wieder stabilisieren.

