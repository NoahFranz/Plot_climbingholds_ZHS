# Plotting_ZHS

Falls du noch keine Erfahrung mit VS Code und Git hast, lese bitte das Beginner README
https://github.com/NoahFranz/Plotting_ZHS/blob/main/BEGINNER_README.md

Ein interaktives Python-Tool zur Verarbeitung und Visualisierung der Klettergriffmessdaten (.lvm) am ZHS der TUM .lvm-Messdaten (ZHS).

## 🔍 Funktionen

- 📊 GUI-Auswahl der darzustellenden Kräfte (Fx, Fy, Fz, Mz)
- 🧲 Separater oder kombinierter Plot von Griff 1 (Rechts) (G1R) und Griff 2 (Links) (G2L)
- 🧹 Optionale Glättung mit Savitzky-Golay-Filter
- 💾 Speichern der Plots mit individuellem Dateinamen und Ausgabeordner
- 🧠 Unterstützt mehrere .lvm-Dateien mit automatischer Gruppierung

## 🖥️ Bedienung

1. Starte das Programm:
   ```bash
   python main.py
   ```

2. Wähle:
   - Welche Plots erstellt werden sollen
   - Welche Griffe und Kräfte angezeigt werden
   - Ob Moment (Mz) getrennt oder kombiniert dargestellt wird
   - Ob Savitzky-Golay-Filter angewendet wird

3. Wähle (optional):
   - Datenordner mit .lvm-Dateien
   - Speicherort und Dateisuffix

## 🔧 Voraussetzungen

- Python 3.9 oder neuer
- Benötigte Pakete:
  ```bash
  pip install -r requirements.txt
  ```

## 📂 Projektstruktur (Auszug)

```
Plotting_ZHS/
├── main.py
├── gui.py
├── loadData.py
├── plotdata.py
├── plotUTILS.py
├── .gitignore
└── requirements.txt
```

## 👨‍🔬 Anwendungsbeispiel

![Beispielplot](docs/beispiel_plot.png)

---

## Individualisierung / Effizienz
In Main.py können der Lese und Speicherort direkt hard-codiert werden, wodurch man die Ordner nicht mehr manuell ausgewählt werden müssen.
    folder_path = data_folder or "/Users/noah/LRZ Sync+Share/MA/ZHS Messungen/Tests/erste Messungen"
    save_folder = gui_save_folder or "/Users/noah/LRZ Sync+Share/MA/Plot_Figures"
Hier einfach den text zwischen "" mit dem gewünschten Pfade erstzen.

## 📬 Kontakt

Bei Fragen oder Feedback gerne melden:  

*noah.franz@pm.me*
