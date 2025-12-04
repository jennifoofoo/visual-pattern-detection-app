# Temporal Cluster Detection
## Identifying Time-Based Patterns in Process Execution

---

## 📋 Was macht Temporal Cluster Detection?

**Ziel:** Identifizierung von zeitlichen und räumlichen Clustern in Event Logs - Perioden mit ungewöhnlich hoher oder niedriger Event-Konzentration.

**Kernidee:** 
- Events treten nicht gleichmäßig verteilt auf → Es gibt "Bursts" und "Lücken"
- Diese Muster können auf Batch-Processing, Schichtwechsel, Systemprobleme oder Ressourcen-Engpässe hinweisen
- **→ View-spezifische Cluster-Erkennung!**

**Anwendungsfälle:**
- 🏥 Krankenhaus: Wann gibt es besonders viele Patientenaufnahmen? (Temporal Bursts)
- 🏭 Produktion: Welche Aktivitäten werden zu bestimmten Zeiten gebündelt? (Activity-Time Clusters)
- 📋 Verwaltung: Wie viele Cases laufen parallel? (Case Parallelism)
- 👥 Personal: Welche Ressourcen arbeiten zu bestimmten Zeiten? (Resource Patterns)

---

## 🧠 Intuition: Warum View-spezifisch?

### ❌ Naiver Ansatz (funktioniert nicht):
```
Alle Events in einem Zeitfenster = Cluster
```
**Problem:** 
- Verschiedene X/Y-Kombinationen haben unterschiedliche Semantik
- `actual_time × activity` → Activity-spezifische Zeit-Cluster
- `actual_time × case_id` → Case Parallelism
- `relative_time × variant` → Variant Timing Patterns

### ✅ View-Aware Ansatz:
```
Für jede X/Y-Kombination:
  1. Bestimme welche Pattern-Typen sinnvoll sind
  2. Wende spezifische Clustering-Algorithmen an
  3. Erkenne nur semantisch bedeutsame Cluster
```

**Beispiel:**
- `actual_time × resource`: Erkenne Resource-Arbeitszeiten (Schichten)
- `relative_time × activity`: Erkenne wann Aktivitäten im Prozess auftreten
- `actual_time × case_id`: Erkenne Case Parallelism (wie viele Cases gleichzeitig)

---

## 🔬 Mathematische Formalisierung

### Algorithmus: DBSCAN (Density-Based Spatial Clustering)

**DBSCAN Parameter:**
- `eps` (ε): Maximale Distanz zwischen zwei Punkten, um im selben Cluster zu sein
- `min_samples`: Minimale Anzahl von Punkten, um einen Cluster zu bilden

**Mathematik:**
```
Für jeden Punkt p:
  - Core Point: wenn ≥ min_samples Punkte in Radius ε
  - Border Point: wenn in ε-Radius eines Core Points, aber selbst kein Core Point
  - Noise: weder Core noch Border Point
```

**Cluster-Definition:**
```
Cluster = {alle Punkte, die von einem Core Point erreichbar sind}
```

### Schritt 1: View-spezifische Pattern-Erkennung

**1. Temporal Bursts** (`actual_time × {activity, resource, case_id}`):
```
X = [t_1, t_2, ..., t_n]  # Zeitpunkte als numerische Werte (Sekunden seit Epoch)
eps = min(5% des Zeitbereichs, 3600 Sekunden)  # Max 1 Stunde
min_samples = 5
```

**2. Activity-Time Clusters** (`{actual_time, relative_time, relative_ratio} × activity`):
```
Für jede Aktivität A:
  X_A = [t_1, t_2, ..., t_k]  # Zeitpunkte für Aktivität A
  eps = auto-berechnet basierend auf Zeitverteilung
  min_samples = max(3, min_cluster_size // 2)
```

**3. Case Parallelism** (`{actual_time, relative_time} × case_id`):
```
Für jeden Case c:
  start_time(c) = min(t | event in case c)
  end_time(c) = max(t | event in case c)

Parallelität(t) = Anzahl Cases mit start_time ≤ t ≤ end_time
max_parallel = max(Parallelität(t) für alle t)
```

**4. Resource Patterns** (`{actual_time, relative_time} × resource`):
```
Für jede Resource R:
  X_R = [t_1, t_2, ..., t_m]  # Zeitpunkte für Resource R
  DBSCAN-Clustering auf X_R
  → Erkenne verschiedene Arbeitszeiten (Schichten)
```

**5. Variant Timing Patterns** (`{relative_time, relative_ratio} × variant`):
```
Für jede Variante V:
  timing_stats(V) = {
    mean: μ_V,
    std: σ_V,
    cv: σ_V / μ_V  # Coefficient of Variation
  }
  
→ Varianten mit unterschiedlichen CV = unterschiedliche Timing-Patterns
```

### Schritt 2: Auto-Parameter-Berechnung

**Temporal Epsilon (eps):**
```
if x_axis == 'actual_time':
    time_range = max(t) - min(t)
    eps = min(time_range * 0.05, 3600)  # 5% des Bereichs oder max 1h
else:
    eps = std(t) * 0.5  # 50% der Standardabweichung
```

**Spatial Epsilon:**
- Wird automatisch basierend auf Y-Achsen-Verteilung berechnet
- Oder manuell konfigurierbar

**Min Cluster Size:**
- Default: 5 Events
- Für Activity-Time: `max(3, min_cluster_size // 2)` (weniger strikt)

### Schritt 3: Cluster-Validierung

**Filter-Kriterien:**
- Mindestens `min_cluster_size` Events pro Cluster
- Noise-Punkte (-1 Label) werden ignoriert
- Nur Cluster mit semantischer Bedeutung werden gespeichert

---

## ⚙️ Implementierungsdetails

### DBSCAN Clustering

**Vorteile:**
- ✅ Keine Vorab-Annahme der Cluster-Anzahl
- ✅ Erkennt Noise-Punkte automatisch
- ✅ Funktioniert mit unregelmäßigen Cluster-Formen
- ✅ Robust gegen Ausreißer

**Nachteile:**
- ⚠️ Sensitiv auf `eps` Parameter
- ⚠️ Schwierig bei unterschiedlichen Cluster-Dichten

### View-Spezifische Anpassungen

**Temporal Bursts:**
- 1D Clustering (nur Zeit-Dimension)
- Auto-eps basierend auf Zeitbereich
- Erkennt Batch-Processing, Schichtwechsel

**Activity-Time Clusters:**
- Pro-Activity Clustering
- Erkennt wann bestimmte Aktivitäten gehäuft auftreten
- Beispiel: "Lab Test" morgens, "Approval" abends

**Case Parallelism:**
- Sweep-Line Algorithmus
- Berechnet maximale gleichzeitige Cases
- Timeline der Parallelität über Zeit

**Resource Patterns:**
- Pro-Resource Clustering
- Erkennt Schichtmuster, Arbeitszeiten
- Identifiziert Ressourcen mit ungewöhnlichen Mustern

**Variant Timing:**
- Statistische Analyse (Mean, Std, CV)
- Vergleicht Timing-Patterns zwischen Varianten
- Erkennt "Fast-Track" vs "Complex" Varianten

---

## 📊 Visualisierung

**Im Dotted Chart:**
- **Temporal Bursts:** Hervorgehobene Zeitbereiche mit hoher Event-Dichte
- **Activity-Time Clusters:** Farbcodierte Cluster pro Aktivität
- **Case Parallelism:** Heatmap der gleichzeitigen Cases
- **Resource Patterns:** Zeitbereiche pro Resource markiert
- **Variant Timing:** Verschiedene Farben für verschiedene Varianten

**Beispiel:**
```
Time:    08:00  09:00  10:00  11:00  12:00
Activity A:  [====]                    [==]
Activity B:        [========]  [====]
Activity C:  [==]              [========]
             ↑                    ↑
         Burst 1              Burst 2
```

---

## 🎯 Aktuelle Implementierung: Stärken

✅ **View-Aware:** Automatische Pattern-Erkennung basierend auf X/Y-Kombination  
✅ **DBSCAN-basiert:** Robustes Density-Based Clustering  
✅ **Auto-Parameter:** Intelligente eps-Berechnung  
✅ **Multi-Pattern:** 5 verschiedene Pattern-Typen  
✅ **Case Parallelism:** Effizienter Sweep-Line Algorithmus  
✅ **Resource-Aware:** Schicht- und Arbeitszeit-Erkennung  

---

## 🚀 Verbesserungsmöglichkeiten

### 1. **Adaptive Epsilon**
**Aktuell:** Statischer eps pro View  
**Verbesserung:**
- Adaptive eps basierend auf lokaler Dichte
- Beispiel: Dichte Clusters → kleinerer eps, spärliche → größerer eps

**Mathematisch:**
```
eps_local(p) = f(k-distance(p, k))
```

### 2. **Multi-Dimensional Clustering**
**Aktuell:** Meist 1D (nur Zeit) oder 2D (Zeit + Y-Achse)  
**Verbesserung:**
- Clustering in höherdimensionalen Räumen
- Beispiel: Zeit + Resource + Activity + Case-Attribute

### 3. **Hierarchical Clustering**
**Aktuell:** Flache Cluster-Struktur  
**Verbesserung:**
- Verschachtelte Cluster (Sub-Cluster innerhalb größerer Cluster)
- Beispiel: Großer Burst enthält mehrere Sub-Bursts

### 4. **Temporal Trends**
**Aktuell:** Statische Cluster-Erkennung  
**Verbesserung:**
- Erkenne Trends über Zeit (Cluster werden größer/kleiner)
- Beispiel: Bursts werden häufiger → System-Überlastung

### 5. **Context-Aware Clustering**
**Aktuell:** Nur Zeit-basiert  
**Verbesserung:**
- Berücksichtige externe Faktoren (Wochentag, Feiertage, Saison)
- Beispiel: Montags gibt es immer Bursts → erwartetes Muster

### 6. **OPTICS Algorithmus**
**Aktuell:** DBSCAN (festes eps)  
**Verbesserung:**
- OPTICS für variable Dichte-Cluster
- Reachability-Plot für Cluster-Visualisierung

### 7. **Cluster-Qualitäts-Metriken**
**Aktuell:** Nur Anzahl der Cluster  
**Verbesserung:**
- Silhouette Score für Cluster-Qualität
- Cohesion & Separation Metriken
- Inter-Cluster vs Intra-Cluster Distanz

### 8. **Incremental Clustering**
**Aktuell:** Recompute alle Cluster bei jedem Run  
**Verbesserung:**
- Update Cluster inkrementell mit neuen Daten
- Effizienter für Streaming Event Logs

### 9. **Anomalie-Erkennung in Clustern**
**Aktuell:** Erkennt nur Cluster  
**Verbesserung:**
- Erkenne Anomalien innerhalb von Clustern
- Beispiel: Cluster hat ungewöhnlich viele Events → möglicher Fehler

### 10. **Interactive Parameter Tuning**
**Aktuell:** Auto-Parameter, manuell schwer anpassbar  
**Verbesserung:**
- UI für eps und min_samples Anpassung
- Live-Vorschau der Cluster-Änderungen

---

## 📈 Beispiel: Konkrete Zahlen

**Event Log:** Hospital Process  
**View:** `actual_time × activity`

**Temporal Burst Detection:**
- Zeitbereich: 2024-01-01 08:00 bis 2024-01-01 18:00 (10 Stunden)
- Auto-eps: `min(10h * 0.05, 1h) = 30 Minuten`
- Min Samples: 5

**Ergebnis:**
- Burst 1: 08:00 - 08:45 (45 min, 23 Events) → Morgen-Rush
- Burst 2: 12:00 - 12:30 (30 min, 18 Events) → Mittagspause-Ende
- Burst 3: 16:00 - 16:20 (20 min, 15 Events) → Schichtwechsel

**Activity-Time Clusters:**
- "Register Patient": 2 Cluster (08:00-10:00, 14:00-16:00)
- "Lab Test": 1 Cluster (09:00-11:00) → Morgendliche Tests
- "Discharge": 1 Cluster (15:00-17:00) → Nachmittags-Entlassungen

---

## 🎓 Zusammenfassung

**Kerninnovation:** View-spezifische Cluster-Erkennung statt generischem Clustering

**Mathematik:** 
- DBSCAN für Density-Based Clustering
- Auto-Parameter-Berechnung
- Sweep-Line für Case Parallelism

**Nächste Schritte:**
- Adaptive Epsilon
- Hierarchical Clustering
- Temporal Trends

---

## 📚 Literatur & Methoden

**Clustering-Algorithmen:**
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- OPTICS (Ordering Points To Identify Clustering Structure)
- K-Means (für Vergleich, aktuell nicht verwendet)

**Process Mining:**
- Temporal Pattern Mining
- Case Parallelism Analysis
- Resource Utilization Patterns

---

*Erstellt für: Visual Pattern Detection in Process Mining*

