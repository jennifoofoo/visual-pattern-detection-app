# Outlier Detection
## Identifying Unusual Events and Cases in Process Execution

---

## 📋 Was macht Outlier Detection?

**Ziel:** Identifizierung ungewöhnlicher Events, Cases oder Ressourcen, die von normalen Prozessmustern abweichen.

**Kernidee:** 
- Nicht alle Events sind "normal" → Manche sind ungewöhnlich in Zeit, Häufigkeit, Sequenz oder Ressourcen-Nutzung
- Outliers können auf Fehler, Sonderfälle, oder interessante Anomalien hinweisen
- **→ Multi-dimensional Outlier Detection!**

**Anwendungsfälle:**
- 🏥 Krankenhaus: Welche Patienten haben ungewöhnlich lange Behandlungszeiten?
- 🏭 Produktion: Welche Fälle haben seltene Aktivitäts-Sequenzen?
- 📋 Verwaltung: Welche Ressourcen haben ungewöhnliche Arbeitsmuster?
- 🔍 Qualitätssicherung: Welche Events treten zu ungewöhnlichen Zeiten auf?

---

## 🧠 Intuition: Warum Multi-Dimensional?

### ❌ Naiver Ansatz (funktioniert nicht):
```
Events mit extremen Werten = Outliers
```
**Problem:** 
- Verschiedene Dimensionen haben unterschiedliche Normalität
- Zeit-Outlier ≠ Häufigkeits-Outlier ≠ Sequenz-Outlier
- Kombination mehrerer Dimensionen gibt besseres Bild

### ✅ Multi-Dimensional Ansatz:
```
Für jede Dimension:
  1. Lerne normale Verteilung
  2. Identifiziere Outliers mit statistischen Methoden
  3. Kombiniere Outliers aus allen Dimensionen
  4. Filtere extreme Outliers (nur Top-K)
```

**Beispiel:**
- **Zeit-Outlier:** Event um 3 Uhr nachts (normal: 8-18 Uhr)
- **Häufigkeits-Outlier:** Aktivität "Emergency" (normal: selten)
- **Sequenz-Outlier:** Transition "A → Z" (normal: "A → B → C")
- **Kombiniert:** Event ist Outlier in mehreren Dimensionen → hohe Priorität

---

## 🔬 Mathematische Formalisierung

### Schritt 1: Multi-Dimensional Outlier Detection

**1. Time-Based Outliers** (wenn Zeit-Daten verfügbar):
```
Für jeden Event e mit Zeit t:
  hour(e) = Stunde von t
  day_of_week(e) = Wochentag von t
  
  rare_hours = {h | count(events mit hour=h) ≤ 5. Perzentil}
  
  if hour(e) in rare_hours:
    mark_as_outlier(e, dimension='time')
```

**2. Case Duration Outliers** (wenn Zeit + Case verfügbar):
```
Für jeden Case c:
  duration(c) = max(t | event in c) - min(t | event in c)
  
  Q1 = 25. Perzentil(durations)
  Q3 = 75. Perzentil(durations)
  IQR = Q3 - Q1
  
  if duration(c) < Q1 - 3×IQR OR duration(c) > Q3 + 3×IQR:
    mark_all_events_in_case_as_outlier(c, dimension='case_duration')
```

**Strikte IQR-Methode:**
- Verwendet **3×IQR** statt 1.5×IQR (klassische Box-Plot Methode)
- Grund: Nur extreme Outliers, nicht alle leichten Abweichungen

**3. Activity Frequency Outliers** (immer möglich):
```
Für jede Aktivität A:
  frequency(A) = Anzahl Events mit activity = A
  total_events = |E|
  
  rare_threshold = max(1, total_events × 0.01)  # 1% Schwellwert
  
  if frequency(A) < rare_threshold:
    mark_all_events_with_activity_as_outlier(A, dimension='activity_frequency')
```

**4. Resource Outliers** (wenn Resource verfügbar):
```
Für jede Resource R:
  workload(R) = Anzahl Events mit resource = R
  
  Q1 = 25. Perzentil(workloads)
  Q3 = 75. Perzentil(workloads)
  IQR = Q3 - Q1
  
  if workload(R) < Q1 - 3×IQR OR workload(R) > Q3 + 3×IQR:
    mark_all_events_with_resource_as_outlier(R, dimension='resource')
```

**5. Sequence Outliers** (wenn Case + Activity verfügbar):
```
Für jeden Case c:
  sequence(c) = [A₁, A₂, ..., Aₙ]  # Aktivitäten in Reihenfolge
  
  transitions = {(Aᵢ, Aᵢ₊₁) | i = 1..n-1}
  
Für jede Transition T:
  frequency(T) = Anzahl Cases mit Transition T
  
  rare_threshold = 1. Perzentil(transition_frequencies)
  
  if frequency(T) ≤ rare_threshold:
    mark_events_in_transition_as_outlier(T, dimension='sequence')
```

**6. Case Complexity Outliers** (wenn Case + Activity verfügbar):
```
Für jeden Case c:
  complexity(c) = {
    event_count: Anzahl Events in c,
    unique_activities: Anzahl verschiedener Aktivitäten,
    transitions: Anzahl Transitionen
  }
  
  # Multi-dimensional Outlier Detection auf complexity(c)
  # Verwendet IQR auf jeder Dimension
```

### Schritt 2: Outlier Score Berechnung

**Für jeden Event e:**
```
score(e) = Σ(1 für jede Dimension, in der e ein Outlier ist)
```

**Beispiel:**
- Event ist Outlier in: time, activity_frequency → score = 2
- Event ist Outlier in: time, case_duration, sequence → score = 3

### Schritt 3: Outlier Kombination

**Kombiniere alle Outliers:**
```
combined_outliers = ∪(outliers[d] für alle Dimensionen d)
```

**Filterung:**
- Wenn > 10% aller Events Outliers → Filtere nur extreme Outliers
- Grund: Zu viele Outliers = Schwellwerte zu niedrig

### Schritt 4: Statistik-Berechnung

**Gesamt-Statistiken:**
```
statistics = {
  total_outliers: |combined_outliers|,
  outlier_percentage: (|combined_outliers| / |E|) × 100,
  max_outlier_score: max(score(e) für alle e),
  cases_with_outliers: Anzahl Cases mit ≥1 Outlier,
  detection_methods_used: Anzahl erfolgreicher Dimensionen
}
```

---

## ⚙️ Implementierungsdetails

### Strikte Schwellwerte

**Warum 3×IQR statt 1.5×IQR?**
- 1.5×IQR: Erkennt ~5% der Daten als Outliers (zu viele False Positives)
- 3×IQR: Erkennt nur extreme Outliers (~0.1% der Daten)
- **Besser für Process Mining:** Wir wollen nur echte Anomalien, nicht leichte Abweichungen

**Warum 1% Schwellwert für Häufigkeit?**
- Aktivitäten die <1% aller Events ausmachen = sehr selten
- Können auf Sonderfälle, Fehler, oder interessante Muster hinweisen

### Dimension-spezifische Anpassungen

**Time Outliers:**
- Benötigt mindestens 10 verschiedene Stunden
- Erkennt nur extrem seltene Stunden (bottom 5%)
- Ignoriert normale Arbeitszeiten

**Case Duration:**
- Filtert Cases mit nur 1 Event (keine Dauer)
- Benötigt mindestens 5 verschiedene Dauer-Werte
- Verwendet 3×IQR für extreme Outliers

**Activity Frequency:**
- Schwellwert: 1% aller Events
- Erkennt sehr seltene Aktivitäten

**Resource:**
- Benötigt mindestens 5 verschiedene Resources
- Verwendet 3×IQR für extreme Workloads

**Sequence:**
- Benötigt mindestens 10 verschiedene Transitionen
- Schwellwert: Bottom 1% (sehr seltene Transitionen)
- Erkennt ungewöhnliche Prozesspfade

**Case Complexity:**
- Multi-dimensional: event_count, unique_activities, transitions
- Verwendet IQR auf jeder Dimension
- Case ist Outlier wenn in ≥1 Dimension Outlier

---

## 📊 Visualisierung

**Im Dotted Chart:**
- **Outlier Events:** Größere, farbige Marker
- **Outlier Cases:** Hervorgehobene Case-Linien
- **Outlier Resources:** Spezielle Markierung
- **Outlier Score:** Farb-Intensität basierend auf Score

**Beispiel:**
```
Normal Events:  ●  ●  ●  ●
Outlier (score=1):  🔴  🔴
Outlier (score=2):  🔴🔴  🔴🔴
Outlier (score=3):  🔴🔴🔴
```

---

## 🎯 Aktuelle Implementierung: Stärken

✅ **Multi-Dimensional:** 6 verschiedene Outlier-Dimensionen  
✅ **Strikte Schwellwerte:** 3×IQR für nur extreme Outliers  
✅ **Adaptiv:** Passt sich an verfügbare Daten an  
✅ **Robust:** Funktioniert auch mit minimalen Daten (case_id + activity)  
✅ **Score-System:** Quantifiziert wie "outlier" ein Event ist  
✅ **Filterung:** Verhindert zu viele False Positives (>10% Outliers)  

---

## 🚀 Verbesserungsmöglichkeiten

### 1. **Isolation Forest**
**Aktuell:** IQR-basierte statistische Methoden  
**Verbesserung:**
- Isolation Forest für komplexe Multi-Dimensional Outliers
- Erkennt Outliers in hochdimensionalen Räumen
- Automatische Feature-Importance

**Mathematisch:**
```
isolation_score(e) = average(path_length(e) in Isolation Trees)
```

### 2. **Local Outlier Factor (LOF)**
**Aktuell:** Globale Schwellwerte  
**Verbesserung:**
- LOF für lokale Outlier-Erkennung
- Berücksichtigt lokale Dichte
- Erkennt Outliers in dichten Regionen

### 3. **Context-Aware Outliers**
**Aktuell:** Statische Schwellwerte  
**Verbesserung:**
- Schwellwerte abhängig von Kontext (Wochentag, Saison, etc.)
- Beispiel: 3 Uhr nachts ist normal am Wochenende, abnormal unter der Woche

### 4. **Temporal Outlier Patterns**
**Aktuell:** Einzelne Events als Outliers  
**Verbesserung:**
- Erkenne temporale Patterns von Outliers
- Beispiel: Outliers treten gehäuft auf → System-Problem

### 5. **Causal Outlier Analysis**
**Aktuell:** Erkennt nur *dass* etwas ein Outlier ist  
**Verbesserung:**
- Erkläre *warum* (welche Faktoren tragen bei?)
- Beispiel: "Case ist Outlier wegen: seltene Transition + lange Dauer + ungewöhnliche Resource"

### 6. **Outlier Severity Levels**
**Aktuell:** Binary (Outlier / Nicht-Outlier)  
**Verbesserung:**
- Severity-Levels: Minor, Moderate, Severe, Extreme
- Basierend auf Abweichung vom Normalen

### 7. **Ensemble Outlier Detection**
**Aktuell:** Einzelne Methoden pro Dimension  
**Verbesserung:**
- Kombiniere mehrere Algorithmen (IQR, Isolation Forest, LOF)
- Voting-System für robustere Erkennung

### 8. **Incremental Outlier Detection**
**Aktuell:** Recompute alle Outliers bei jedem Run  
**Verbesserung:**
- Update Outlier-Liste inkrementell mit neuen Daten
- Effizienter für Streaming Event Logs

### 9. **Outlier Explanation**
**Aktuell:** Nur Score, keine Erklärung  
**Verbesserung:**
- Automatische Erklärungen: "Event ist Outlier weil: seltene Aktivität (0.5%), ungewöhnliche Zeit (3 Uhr), seltene Transition"
- AI-basierte Erklärungen (z.B. mit Ollama)

### 10. **Interactive Threshold Tuning**
**Aktuell:** Feste Schwellwerte (3×IQR, 1%)  
**Verbesserung:**
- UI für Schwellwert-Anpassung
- Live-Vorschau der Outlier-Änderungen
- Sensitivitäts-Slider (Strict / Moderate / Lenient)

---

## 📈 Beispiel: Konkrete Zahlen

**Event Log:** Hospital Process  
**Total Events:** 10,000

**Time Outliers:**
- Seltene Stunden: 2-4 Uhr (bottom 5%)
- Outliers: 45 Events (0.45%)

**Case Duration Outliers:**
- Q1 = 2 Stunden, Q3 = 8 Stunden, IQR = 6 Stunden
- Threshold: Q1 - 3×IQR = -16h (unmöglich), Q3 + 3×IQR = 26 Stunden
- Outliers: Cases > 26 Stunden → 12 Cases (1.2%)

**Activity Frequency Outliers:**
- Schwellwert: 10,000 × 0.01 = 100 Events
- Seltene Aktivitäten: "Emergency Surgery" (15 Events), "Code Blue" (8 Events)
- Outliers: 23 Events (0.23%)

**Sequence Outliers:**
- Seltene Transitionen: Bottom 1% (≤ 5 Fälle)
- Beispiel: "Discharge → Register" (nur 3 Fälle, normal: "Register → ...")
- Outliers: 156 Events (1.56%)

**Kombiniert:**
- Total Outliers: 234 Events (2.34%)
- Max Score: 3 (Outlier in 3 Dimensionen)
- Cases betroffen: 45 Cases (4.5%)

---

## 🎓 Zusammenfassung

**Kerninnovation:** Multi-dimensionale Outlier-Erkennung mit strikten Schwellwerten

**Mathematik:** 
- IQR-basierte Outlier Detection (3×IQR)
- Perzentil-basierte Schwellwerte
- Multi-dimensional Score-System

**Nächste Schritte:**
- Isolation Forest für komplexe Outliers
- Context-Aware Schwellwerte
- Causal Analysis für Erklärungen

---

## 📚 Literatur & Methoden

**Statistische Methoden:**
- Interquartile Range (IQR) für Outlier Detection
- Perzentile für robuste Schwellwerte
- Multi-dimensional Outlier Scoring

**Machine Learning:**
- Isolation Forest (für zukünftige Implementierung)
- Local Outlier Factor (LOF)
- Ensemble Methods

**Process Mining:**
- Sequence-based Outlier Detection
- Case Complexity Analysis
- Resource Behavior Analysis

---

*Erstellt für: Visual Pattern Detection in Process Mining*

