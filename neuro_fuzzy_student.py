"""
==============================================================
 HYBRID NEURO-FUZZY SYSTEM FOR STUDENT PERFORMANCE PREDICTION
 Author  : Sourav
 Course  : Soft Computing (MUJ)
 Method  : ANFIS-inspired (Adaptive Neuro-Fuzzy Inference System)
           Fuzzy Layer → Neural Learning via Backpropagation
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1.  FUZZY MEMBERSHIP FUNCTIONS
# ─────────────────────────────────────────────────────────────

def trimf(x, a, b, c):
    """Triangular membership function."""
    x = np.asarray(x, dtype=float)
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9),
                                     (c - x) / (c - b + 1e-9)))

def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    x = np.asarray(x, dtype=float)
    return np.maximum(0, np.minimum(
        np.minimum((x - a) / (b - a + 1e-9), 1),
        (d - x) / (d - c + 1e-9)))

# Membership function parameters (learnable in full ANFIS)
MF_PARAMS = {
    'attendance': {
        'Low':    ('trapmf', [0,   0,  50, 65]),
        'Medium': ('trimf',  [55, 70,  85]),
        'High':   ('trapmf', [75, 88, 100, 100]),
    },
    'assignment': {
        'Low':    ('trapmf', [0,   0,  35, 50]),
        'Medium': ('trimf',  [40, 55,  70]),
        'High':   ('trapmf', [60, 75, 100, 100]),
    },
    'test': {
        'Low':    ('trapmf', [0,   0,  35, 50]),
        'Medium': ('trimf',  [40, 55,  70]),
        'High':   ('trapmf', [60, 75, 100, 100]),
    },
}

def compute_mf(value, feature):
    """Return [Low, Medium, High] membership degrees for a feature."""
    mfs = []
    for term, (fn, params) in MF_PARAMS[feature].items():
        if fn == 'trimf':
            mfs.append(trimf(value, *params))
        else:
            mfs.append(trapmf(value, *params))
    return np.array(mfs)   # shape (3,)

# ─────────────────────────────────────────────────────────────
# 2.  FUZZY RULE BASE  (27 rules: 3^3 full coverage)
# ─────────────────────────────────────────────────────────────
#  Indices: 0=Low, 1=Medium, 2=High
#  Output : 0=Poor, 1=Average, 2=Good

RULES = {}   # (att_idx, asn_idx, tst_idx) -> output_class

def _set(a, b, c, out): RULES[(a, b, c)] = out

# High test → Good unless everything else is low
_set(2, 2, 2, 2); _set(2, 2, 1, 2); _set(2, 1, 2, 2)
_set(1, 2, 2, 2); _set(2, 1, 1, 1); _set(1, 2, 1, 1)
_set(1, 1, 2, 1); _set(2, 2, 0, 1); _set(2, 0, 2, 1)
_set(0, 2, 2, 1); _set(2, 0, 1, 0); _set(2, 0, 0, 0)
_set(1, 1, 1, 1); _set(1, 1, 0, 0); _set(1, 0, 1, 0)
_set(1, 0, 0, 0); _set(0, 1, 1, 0); _set(0, 1, 0, 0)
_set(0, 0, 1, 0); _set(0, 0, 0, 0); _set(0, 2, 1, 1)
_set(0, 2, 0, 0); _set(0, 1, 2, 1); _set(0, 0, 2, 1)
_set(1, 0, 2, 1); _set(2, 1, 0, 1); _set(1, 2, 0, 1)

assert len(RULES) == 27, "Need exactly 27 rules"

# ─────────────────────────────────────────────────────────────
# 3.  FUZZY INFERENCE ENGINE  (Mamdani-style crisp output)
# ─────────────────────────────────────────────────────────────

def fuzzy_infer(att, asn, tst):
    """
    Returns a soft 3-vector [μ_Poor, μ_Average, μ_Good]
    aggregated from all 27 rules (used as neural network input).
    """
    mu_att = compute_mf(att, 'attendance')
    mu_asn = compute_mf(asn, 'assignment')
    mu_tst = compute_mf(tst, 'test')

    agg = np.zeros(3)  # [Poor, Average, Good]
    for (ai, bi, ci), out_class in RULES.items():
        strength = mu_att[ai] * mu_asn[bi] * mu_tst[ci]   # product T-norm
        agg[out_class] = max(agg[out_class], strength)

    if agg.sum() < 1e-9:
        agg = np.array([1.0, 0.0, 0.0])  # default: Poor
    return agg   # not normalised → NN sees raw firing strengths

# ─────────────────────────────────────────────────────────────
# 4.  NEURAL NETWORK CLASSIFIER  (2-layer MLP, numpy)
# ─────────────────────────────────────────────────────────────

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def relu(z):     return np.maximum(0, z)
def drelu(z):    return (z > 0).astype(float)

class MLP:
    """Small 2-hidden-layer MLP trained with mini-batch SGD."""
    def __init__(self, in_dim, h1=16, h2=8, out_dim=3, lr=0.05, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.3, (in_dim, h1))
        self.b1 = np.zeros(h1)
        self.W2 = rng.normal(0, 0.3, (h1, h2))
        self.b2 = np.zeros(h2)
        self.W3 = rng.normal(0, 0.3, (h2, out_dim))
        self.b3 = np.zeros(out_dim)
        self.lr  = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.out = softmax(self.z3)
        self.X   = X
        return self.out

    def backward(self, y_oh):
        N   = y_oh.shape[0]
        dz3 = (self.out - y_oh) / N
        dW3 = self.a2.T @ dz3
        db3 = dz3.sum(0)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * drelu(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * drelu(self.z1)
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(0)
        for p, g in [(self.W1,dW1),(self.b1,db1),(self.W2,dW2),
                     (self.b2,db2),(self.W3,dW3),(self.b3,db3)]:
            p -= self.lr * g

    def cross_entropy(self, y_oh):
        return -np.mean(np.sum(y_oh * np.log(self.out + 1e-9), axis=1))

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ─────────────────────────────────────────────────────────────
# 5.  HYBRID PIPELINE
# ─────────────────────────────────────────────────────────────

class HybridNeuroFuzzy:
    """
    Architecture
    ─────────────────────────────────────────────────────────
    Input (att, asn, tst)
        │
    [FUZZY LAYER]  → 27-rule Mamdani → 3-D firing vector
        │               +
    [FEATURE CONCAT] → [att/100, asn/100, tst/100] → 6-D input
        │
    [MLP]  16 → 8 → 3 (softmax) → {Poor, Average, Good}
        │
    [BACKPROP] ← cross-entropy loss ← ground truth labels
    ─────────────────────────────────────────────────────────
    The fuzzy layer pre-processes raw scores into linguistic
    firing strengths, giving the NN a semantically richer
    input than raw numbers alone.
    """

    def __init__(self, lr=0.05):
        self.nn = MLP(in_dim=6, h1=16, h2=8, out_dim=3, lr=lr)

    def _extract(self, X):
        """X: (N,3) = [attendance, assignment, test], all 0-100."""
        feats = []
        for row in X:
            att, asn, tst = row
            fuzz = fuzzy_infer(att, asn, tst)           # 3-D fuzzy vector
            raw  = np.array([att, asn, tst]) / 100.0    # 3-D normalised raw
            feats.append(np.concatenate([fuzz, raw]))   # 6-D
        return np.array(feats)

    def fit(self, X, y, epochs=300, batch=32, verbose=True):
        N    = len(X)
        Xf   = self._extract(X)
        y_oh = np.eye(3)[y]
        losses = []
        idx  = np.arange(N)
        for ep in range(1, epochs+1):
            np.random.shuffle(idx)
            for i in range(0, N, batch):
                bi = idx[i:i+batch]
                self.nn.forward(Xf[bi])
                self.nn.backward(y_oh[bi])
            self.nn.forward(Xf)
            loss = self.nn.cross_entropy(y_oh)
            losses.append(loss)
            if verbose and ep % 50 == 0:
                acc = np.mean(self.nn.predict(Xf) == y)
                print(f"Epoch {ep:4d}  Loss: {loss:.4f}  Acc: {acc*100:.1f}%")
        return losses

    def predict(self, X):
        Xf = self._extract(X)
        return self.nn.predict(Xf)

    def predict_proba(self, X):
        Xf = self._extract(X)
        return self.nn.forward(Xf)

# ─────────────────────────────────────────────────────────────
# 6.  SYNTHETIC DATASET  (500 labelled students)
# ─────────────────────────────────────────────────────────────

def generate_dataset(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    labels = {0: 'Poor', 1: 'Average', 2: 'Good'}
    for _ in range(n):
        cls = rng.choice([0, 1, 2], p=[0.30, 0.40, 0.30])
        if cls == 0:      # Poor
            att = rng.uniform(30, 65)
            asn = rng.uniform(20, 55)
            tst = rng.uniform(20, 55)
        elif cls == 1:    # Average
            att = rng.uniform(55, 85)
            asn = rng.uniform(45, 72)
            tst = rng.uniform(45, 72)
        else:             # Good
            att = rng.uniform(75, 100)
            asn = rng.uniform(68, 100)
            tst = rng.uniform(68, 100)
        att = np.clip(att + rng.normal(0, 4), 0, 100)
        asn = np.clip(asn + rng.normal(0, 4), 0, 100)
        tst = np.clip(tst + rng.normal(0, 4), 0, 100)
        X.append([att, asn, tst])
        y.append(cls)
    return np.array(X), np.array(y)

# ─────────────────────────────────────────────────────────────
# 7.  TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 58)
    print("  HYBRID NEURO-FUZZY STUDENT PERFORMANCE SYSTEM")
    print("=" * 58)

    X, y = generate_dataset(500)
    split = int(0.8 * len(X))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:],  y[split:]

    model = HybridNeuroFuzzy(lr=0.05)
    print("\n--- Training (300 epochs) ---")
    losses = model.fit(X_tr, y_tr, epochs=300, batch=32)

    preds = model.predict(X_te)
    acc   = np.mean(preds == y_te)
    print(f"\nTest accuracy : {acc*100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_te, preds,
          target_names=['Poor','Average','Good']))

    # ── single-sample demo ───────────────────────────────────
    demo = np.array([[72, 68, 74]])
    prob = model.predict_proba(demo)[0]
    cls  = ['Poor','Average','Good'][np.argmax(prob)]
    print(f"\nDemo student (att=72, asn=68, tst=74)")
    print(f"  Poor={prob[0]:.3f}  Average={prob[1]:.3f}  Good={prob[2]:.3f}")
    print(f"  Predicted: {cls}")

    # ── plots ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig)

    # MF plots
    x = np.linspace(0, 100, 300)
    colors = {'Low':'#E24B4A','Medium':'#BA7517','High':'#1D9E75'}
    for fi, feat in enumerate(['attendance','assignment','test']):
        ax = fig.add_subplot(gs[0, fi])
        for term, (fn, p) in MF_PARAMS[feat].items():
            y_ = trimf(x,*p) if fn=='trimf' else trapmf(x,*p)
            ax.plot(x, y_, label=term, color=colors[term], lw=2)
        ax.set_title(feat.capitalize(), fontweight='bold')
        ax.set_xlabel('Score (0-100)'); ax.set_ylabel('μ')
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.1); ax.grid(alpha=0.3)

    # Loss curve
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(losses, color='#534AB7', lw=1.5)
    ax.set_title('Training Loss', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy')
    ax.grid(alpha=0.3)

    # Confusion matrix
    ax = fig.add_subplot(gs[1, 0:2])
    cm = confusion_matrix(y_te, preds)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(['Poor','Average','Good'])
    ax.set_yticklabels(['Poor','Average','Good'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix', fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i,j], ha='center', va='center',
                    color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=14)
    plt.colorbar(im, ax=ax)

    # Probability bar for demo
    ax = fig.add_subplot(gs[1, 2:])
    bars = ax.bar(['Poor','Average','Good'], prob,
                  color=['#E24B4A','#BA7517','#1D9E75'], edgecolor='white')
    ax.set_ylim(0, 1); ax.set_ylabel('Probability')
    ax.set_title('Demo Student Output Probabilities', fontweight='bold')
    for bar, v in zip(bars, prob):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f'{v:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Hybrid Neuro-Fuzzy Student Performance System', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('neuro_fuzzy_output.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: neuro_fuzzy_output.png")
    plt.show()
