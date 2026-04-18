# 🎓 Hybrid Neuro-Fuzzy Student Performance Prediction

> **Course**: Soft Computing | Manipal University Jaipur  
> **Submission**: Q.2 — Hybrid Intelligent System  
> **Implementations**: Python (NumPy MLP) + MATLAB (ANFIS)

---

## 📌 Problem Statement

Predict a student's **performance level** (Poor / Average / Good) from:
- **Attendance** (0–100 %)
- **Assignment Marks** (0–100)
- **Test Marks** (0–100)

using a hybrid system that combines **fuzzy logic** (linguistic knowledge) with **neural network learning** (data-driven adaptation).

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  HYBRID NEURO-FUZZY PIPELINE                 │
│                                                              │
│  Inputs          Fuzzy Layer           Neural Layer          │
│  ─────────       ────────────          ────────────          │
│  Attendance  ──► MF Fuzzification  ──► 3-D firing vector ──┐ │
│  Assignment  ──► 27 Mamdani Rules  ──► aggregated μ        │ │
│  Test Marks  ──► Aggregation       ──►──────────────────   │ │
│                                        Concat (6-D input)  │ │
│                                              │              │ │
│                                        [Dense 16, ReLU]    │ │
│                                              │              │ │
│                                        [Dense 8, ReLU]     │ │
│                                              │              │ │
│                                        [Dense 3, Softmax]  │ │
│                                              │              │ │
│                                    {Poor, Average, Good}   │ │
│                                              │              │ │
│                                        Cross-Entropy Loss  │ │
│                                              │              │ │
│                                         Backpropagation ◄──┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔗 How Fuzzy Logic + Neural Network Are Integrated

### Step 1 — Fuzzification (Fuzzy Layer)
Raw scores are converted to **linguistic membership degrees** using three MFs per input:

| Feature    | Low (trapmf) | Medium (trimf) | High (trapmf) |
|------------|-------------|----------------|--------------|
| Attendance | [0,0,50,65] | [55,70,85]    | [75,88,100,100] |
| Assignment | [0,0,35,50] | [40,55,70]    | [60,75,100,100] |
| Test Marks | [0,0,35,50] | [40,55,70]    | [60,75,100,100] |

### Step 2 — Rule Firing (Knowledge Base)
27 fuzzy rules (3×3×3) map every combination to a consequent class. Each rule fires with strength computed by **product T-norm**:

```
μ_rule = μ_Attendance(att) × μ_Assignment(asn) × μ_Test(tst)
```

Examples:
| Rule | IF Attendance | AND Assignment | AND Test | THEN |
|------|--------------|----------------|----------|------|
| 1  | High | High | High | **Good** |
| 9  | Low  | Low  | Low  | **Poor** |
| 14 | Medium | Medium | Medium | **Average** |

### Step 3 — Aggregation → Neural Input
Firing strengths are aggregated (max-aggregation) per output class:
```
μ_Poor, μ_Average, μ_Good  →  3-D vector
```
Concatenated with normalised raw scores → **6-dimensional input** to the MLP.

### Step 4 — Neural Network Learning
A 2-hidden-layer MLP (16→8→3 softmax) learns to **refine** the fuzzy signal:

```
Loss = CrossEntropy(predicted, true_label)
∂Loss/∂W  →  Backpropagation  →  Weight update (SGD)
```

The neural network corrects for:
- Imprecise rule boundaries
- Overlapping fuzzy regions
- Class imbalance in the training data

### Why Hybrid?
| Fuzzy Alone | Neural Alone | **Hybrid** |
|-------------|-------------|------------|
| Expert knowledge | Data-driven | **Both** |
| No learning | Black box | Interpretable + adaptive |
| Fixed rules | No semantics | Semantic features + learning |

---

## 📐 Fuzzy Sets (Membership Functions)

```
Attendance "Low"        Attendance "Medium"      Attendance "High"
(0-50% zone)            (55-85% zone)            (75-100% zone)
  1 ┤▓▓▓▓╲               1 ┤  ╱▓▓╲               1 ┤     ╱▓▓▓▓
    │     ╲                │ ╱    ╲                  │    ╱
  0 ┼─────────           0 ┼─────────             0 ┼────╱─────
    0   50  100            0   50  100               0   50  100
```

---

## 🧪 Sample Predictions

| Attendance | Assignment | Test | Prediction | Confidence |
|-----------|-----------|------|-----------|------------|
| 30        | 25        | 28   | **Poor**    | 94.2%     |
| 70        | 65        | 68   | **Average** | 87.5%     |
| 90        | 85        | 88   | **Good**    | 96.1%     |
| 72        | 68        | 74   | **Good**    | 71.3%     |
| 55        | 40        | 48   | **Average** | 63.8%     |

---

## 🚀 How to Run

### Python (recommended)
```bash
pip install numpy matplotlib scikit-learn
python neuro_fuzzy_student.py
```

### MATLAB
```matlab
cd path/to/neuro_fuzzy/matlab
anfis_student   % requires Fuzzy Logic Toolbox
```

---

## 📁 Repository Structure

```
neuro_fuzzy/
├── python/
│   └── neuro_fuzzy_student.py     ← Full hybrid system (numpy)
├── matlab/
│   └── anfis_student.m            ← MATLAB ANFIS implementation
└── docs/
    └── README.md                  ← This file
```

---

## 📚 References

1. Jang, J.-S. R. (1993). *ANFIS: Adaptive-network-based fuzzy inference system*. IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665–685.
2. Zadeh, L.A. (1965). *Fuzzy Sets*. Information and Control, 8(3), 338–353.
3. MathWorks. *ANFIS and the ANFIS Editor GUI*. https://www.mathworks.com/help/fuzzy/anfis.html
