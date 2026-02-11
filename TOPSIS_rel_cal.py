import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fix Chinese font rendering issues (recommended)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Raw data
services = ["Physiotherapy & Wellness", "Medical Assistance", "Meal Assistance", "Smart Elderly Care", "Emergency Assistance", "Social Interaction", "Information Push", "Cultural & Educational Interests"]
SI = [0.59, 0.77, 0.78, 0.59, 0.64, 0.59, 0.83, 0.58]
DSI = [-0.67, -0.52, -0.53, -0.69, -0.67, -0.59, -0.42, -0.51]
loading = [0.813, 0.807, 0.844, 0.790, 0.772, 0.770, 0.816, 0.832]
DSI_pos = [-x for x in DSI]

# EWM-TOPSIS calculation
X = np.array([SI, DSI_pos, loading]).T
Z = X / np.sqrt((X ** 2).sum(axis=0))
P = Z / Z.sum(axis=0)
P = np.where(P == 0, 1e-6, P)
k = 1 / np.log(len(services))
E = -k * (P * np.log(P)).sum(axis=0)
d = 1 - E
w = d / d.sum()
V = Z * w
V_plus = V.max(axis=0)
V_minus = V.min(axis=0)
D_plus = np.sqrt(((V - V_plus) ** 2).sum(axis=1))
D_minus = np.sqrt(((V - V_minus) ** 2).sum(axis=1))
C = D_minus / (D_plus + D_minus)

# Ranking results
df = pd.DataFrame({
    "Service Item": services,
    "SI": SI,
    "DSI": DSI,
    "Factor Loading": loading,
    "DSI (Positive)": DSI_pos,
    "Score": C,
    "D+": D_plus,
    "D-": D_minus
}).sort_values(by="Score", ascending=False).reset_index(drop=True)

# 1. Radar chart of weights
labels = ["SI", "DSI", "Factor Loading"]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
weights = list(w) + [w[0]]

fig1, ax1 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax1.plot(angles, weights, color='navy', linewidth=2)
ax1.fill(angles, weights, color='skyblue', alpha=0.4)
ax1.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=16)
#ax1.set_title("Radar Chart of Indicator Weights", fontsize=16)
ax1.tick_params(axis='y',labelsize=14)

# 2. Bar chart of TOPSIS scores
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.barh(df["Service Item"], df["Score"], color='mediumseagreen')
for i, v in enumerate(df["Score"]):
    ax2.text(v + 0.005, i, f"{v:.4f}", va='center', fontsize=11)
#ax2.set_title("Bar Chart of TOPSIS Scores", fontsize=16)
ax2.set_xlabel("Relative Closeness Score", fontsize=12)
ax2.set_ylabel("Service Demands", fontsize=12)
ax2.set_xlim(0, 0.7)  # maximum x-axis value
ax2.tick_params(axis='x', labelsize=13)
ax2.tick_params(axis='y', labelsize=11)
ax2.invert_yaxis()


# 3. Radar chart of TOPSIS ranking
angles3 = np.linspace(0, 2 * np.pi, len(services), endpoint=False).tolist()
angles3 += angles3[:1]
scores = list(C) + [C[0]]

fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax3.plot(angles3, scores, color='green', linewidth=2)
ax3.fill(angles3, scores, color='lightgreen', alpha=0.4)
ax3.set_thetagrids(np.degrees(angles3[:-1]), services, fontsize=14)
ax3.xaxis.set_tick_params(pad=20)  # set the distance of angular labels from the center
# ✅ Set the maximum of the y-axis so label positions shrink inward naturally
ax3.set_ylim(0, max(scores) + 0.05)
#ax3.set_title("TOPSIS Ranking Radar Chart", fontsize=16)
ax3.tick_params(axis='y', labelsize=12)

# Add score labels at each vertex
for angle, score in zip(angles3[:-1], scores[:-1]):
    ax3.text(angle, score + 0.03, f"{score:.2f}", ha='center', va='center', fontsize=12)


plt.tight_layout()
plt.show()
