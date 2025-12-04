# Process-Aware Gap Detection
## Detecting Abnormal Delays in Process Execution

---

## 📋 Was macht Gap Detection?

**Ziel:** Identifizierung ungewöhnlich langer Wartezeiten zwischen aufeinanderfolgenden Aktivitäten in einem Prozess.

**Kernidee:** 
- Nicht alle Gaps sind gleich → Ein 2-Stunden-Gap zwischen "Register" → "Review" ist normal
- Aber ein 2-Stunden-Gap zwischen "Review" → "Approve" könnte abnormal sein
- **→ Transition-spezifische Normalität lernen!**

**Anwendungsfälle:**
- 🏥 Krankenhaus: Welche Ressourcen haben ungewöhnliche Wartezeiten?
- 🏭 Produktion: Wo gibt es Bottlenecks zwischen Aktivitäten?
- 📋 Verwaltung: Welche Prozessschritte dauern länger als erwartet?

---

## 🧠 Intuition: Warum Transition-spezifisch?

### ❌ Naiver Ansatz (funktioniert nicht):
```
Alle Gaps > 1 Stunde = abnormal
```
**Problem:** 
- "Register → Review" dauert normalerweise 2 Stunden → 1h wäre zu kurz!
- "Review → Approve" dauert normalerweise 10 Minuten → 1h wäre abnormal!

### ✅ Process-Aware Ansatz:
```
Für jede Transition (A → B):
  1. Lerne normale Gap-Dauer aus historischen Daten
  2. Berechne statistischen Schwellwert
  3. Erkenne Gaps, die diesen Schwellwert überschreiten
```

**Beispiel:**
- "Register → Review": Median = 2h, Threshold = 4h → Gap von 5h = abnormal ✅
- "Review → Approve": Median = 10min, Threshold = 20min → Gap von 5h = abnormal ✅

---

## 🔬 Mathematische Formalisierung

### Schritt 1: Gap-Extraktion
Für jeden Case `c` und aufeinanderfolgende Events `e_i`, `e_{i+1}`:

```
gap = t(e_{i+1}) - t(e_i)
transition = activity(e_i) → activity(e_{i+1})
```

**Eigenschaften:**
- Case-aware: Nur Gaps innerhalb desselben Cases
- Transition-aware: Speichere welche Aktivitäten beteiligt sind

### Schritt 2: Statistische Normality-Berechnung

Für jede Transition `T = A → B`:

**1. Sammle alle Gap-Dauern:**
```
D_T = {gap_1, gap_2, ..., gap_n}  für alle Gaps mit Transition T
```

**2. Berechne Deskriptive Statistiken:**
```
median_T = median(D_T)
Q1_T = 25. Perzentil(D_T)
Q3_T = 75. Perzentil(D_T)
IQR_T = Q3_T - Q1_T
P95_T = 95. Perzentil(D_T)
```

**3. Berechne Threshold (Anomalie-Schwelle):**
```
threshold_T = max(P95_T, Q3_T + 1.5 × IQR_T)
```

**Intuition:**
- `P95_T`: 95% aller Gaps sind kürzer → 5% sind länger (potentiell abnormal)
- `Q3_T + 1.5×IQR_T`: Box-Plot Outlier-Definition (klassische statistische Methode)
- `max(...)`: Verwende die konservativere (höhere) Schwelle

**4. Minimum Sample Size:**
```
if |D_T| < 5:
    skip transition T  # Nicht genug Daten für statistische Aussage
```

### Schritt 3: Abnormal Gap Detection

Für jeden Gap `g` mit Transition `T`:

```
if gap_duration(g) > threshold_T:
    severity(g) = gap_duration(g) / threshold_T
    mark_as_abnormal(g)
```

**Severity Interpretation:**
- `1.0 - 1.5`: Leichte Abweichung
- `1.5 - 2.0`: Moderate Anomalie
- `2.0 - 5.0`: Signifikante Anomalie
- `> 5.0`: Extreme Anomalie

---

## ⚙️ Processing Time Berücksichtigung (Optional)

**Problem:** Gap = Processing Time + Waiting Time

**Lösung:** Schätze Processing Time pro Aktivität:

```
processing_time(A) = median({
    duration(e_i → e_{i+1}) | 
    activity(e_i) = activity(e_{i+1}) = A
})
```

**Wartezeit-Berechnung:**
```
waiting_time = gap_duration - processing_time(activity_from)
```

**Vorteil:** 
- Fokus auf **Wartezeiten** statt Gesamt-Gap
- Identifiziert echte Bottlenecks (nicht nur langsame Aktivitäten)

**Einschränkung:**
- Funktioniert nur für `actual_time` und `relative_time` (absolute Zeitwerte)
- Benötigt mindestens 3 Samples pro Aktivität für median

---

## 📊 Visualisierung

**Im Dotted Chart:**
- Rote Rechtecke über abnormalen Gaps
- Position: Y-Achse = FROM-Activity (wo die Wartezeit passiert)
- X-Achse: Von `x_start` bis `x_end` (Gap-Dauer)
- Opacity: Basierend auf Severity

**Beispiel:**
```
Resource R1:  [====]  ← Normaler Gap
Resource R2:  [============]  ← Abnormaler Gap (rot markiert)
Resource R3:  [====]
```

---

## 🎯 Aktuelle Implementierung: Stärken

✅ **Process-Aware:** Transition-spezifische Thresholds  
✅ **Robust:** Minimum Sample Size Check (≥5 Samples)  
✅ **Statistisch fundiert:** IQR + P95 Kombination  
✅ **Severity-Berechnung:** Quantifiziert wie abnormal ein Gap ist  
✅ **Processing Time:** Optional für `actual_time`/`relative_time`  
✅ **Visualisierung:** Stabile Y-Position Berechnung (kategorisch + numerisch)  

---

## 🚀 Verbesserungsmöglichkeiten

### 1. **Adaptive Thresholds**
**Aktuell:** Statischer Threshold pro Transition  
**Verbesserung:** 
- Berücksichtige Tageszeit, Wochentag, Saison
- Beispiel: "Register → Review" dauert montags länger → unterschiedliche Thresholds

**Mathematisch:**
```
threshold_T(t) = f(transition_T, time_features(t))
```

### 2. **Context-Aware Processing Time**
**Aktuell:** Median über alle Fälle  
**Verbesserung:**
- Processing Time abhängig von Case-Attributen
- Beispiel: Komplexe Fälle brauchen länger → unterschiedliche Processing Times

### 3. **Multi-Transition Patterns**
**Aktuell:** Nur einzelne Transitions (A → B)  
**Verbesserung:**
- Erkenne Patterns über mehrere Transitions
- Beispiel: "A → B → C" Sequenz hat immer lange Gaps

### 4. **Resource-Specific Normality**
**Aktuell:** Transition-spezifisch, aber nicht Resource-spezifisch  
**Verbesserung:**
- Lerne Normality pro Resource + Transition
- Beispiel: "Register → Review" bei Resource R1 dauert länger als bei R2

**Mathematisch:**
```
threshold_{T,R} = f(transition_T, resource_R)
```

### 5. **Temporal Trends**
**Aktuell:** Statische Thresholds (basierend auf allen historischen Daten)  
**Verbesserung:**
- Berücksichtige Trends über Zeit
- Beispiel: Gaps werden generell länger → Threshold sollte sich anpassen

### 6. **Uncertainty Quantification**
**Aktuell:** Binary (abnormal / normal)  
**Verbesserung:**
- Konfidenz-Intervall für Threshold
- Beispiel: "Gap ist abnormal mit 95% Konfidenz"

### 7. **Causal Analysis**
**Aktuell:** Erkennt nur *dass* ein Gap abnormal ist  
**Verbesserung:**
- Erkläre *warum* (welche Faktoren tragen bei?)
- Beispiel: "Gap ist abnormal wegen: hohe Auslastung von Resource R1, viele parallele Cases"

### 8. **Incremental Learning**
**Aktuell:** Recompute alle Thresholds bei jedem Run  
**Verbesserung:**
- Update Thresholds inkrementell mit neuen Daten
- Effizienter für große Event Logs

### 9. **Handling Missing Data**
**Aktuell:** Überspringt Transitions mit <5 Samples  
**Verbesserung:**
- Transfer Learning: Nutze ähnliche Transitions für Schätzung
- Beispiel: "A → B" hat nur 3 Samples, aber "A → C" hat 100 → verwende ähnliche Threshold

### 10. **Interactive Threshold Tuning**
**Aktuell:** Feste Formel (max(P95, Q3+1.5×IQR))  
**Verbesserung:**
- User kann Threshold-Sensitivität anpassen
- Beispiel: "Stricter" (höhere Thresholds) vs "Lenient" (niedrigere Thresholds)

---

## 📈 Beispiel: Konkrete Zahlen

**Event Log:** Hospital Process  
**Transition:** "Register Patient" → "First Examination"

**Statistiken:**
- Anzahl Samples: 150
- Median: 45 Minuten
- Q1: 30 Minuten
- Q3: 60 Minuten
- IQR: 30 Minuten
- P95: 90 Minuten

**Threshold:**
```
threshold = max(90, 60 + 1.5 × 30)
          = max(90, 105)
          = 105 Minuten
```

**Ergebnis:**
- Gap von 120 Minuten → Severity = 120/105 = 1.14 → Leichte Abweichung
- Gap von 200 Minuten → Severity = 200/105 = 1.90 → Moderate Anomalie
- Gap von 500 Minuten → Severity = 500/105 = 4.76 → Signifikante Anomalie

---

## 🎓 Zusammenfassung

**Kerninnovation:** Transition-spezifische Normalität statt globaler Thresholds

**Mathematik:** 
- IQR-basierte Outlier-Detection
- Perzentil-basierte Schwellwerte
- Severity-Quantifizierung

**Nächste Schritte:**
- Adaptive Thresholds (zeitabhängig)
- Resource-spezifische Normality
- Causal Analysis für Erklärungen

---

## 📚 Literatur & Methoden

**Statistische Methoden:**
- Interquartile Range (IQR) für Outlier Detection
- Perzentile für robuste Schwellwerte
- Median statt Mean (robust gegen Ausreißer)

**Process Mining:**
- Transition-based Analysis
- Case-aware Gap Extraction
- Activity-aware Semantics

---

*Erstellt für: Visual Pattern Detection in Process Mining*

