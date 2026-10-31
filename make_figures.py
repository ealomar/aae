"""Manuscript figures 1-7, drawn at their printed size (column = 3.0 in, full width = 6.0 in)
so that no text is smaller than 7.5 pt in print. 600 dpi, no titles inside the images.
Figures 4-7 are computed from the re-analysis output and match Tables 9, 12 and 14 (R2 numbering)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COL, FULL = 3.0, 6.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": "#333333",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.color": "#111111", "ytick.color": "#111111"})
NAVY, LIGHT, MID, INK, ORANGE, ORANGE_F = "#2F5D8A", "#C9D7E6", "#7FA1C4", "#111111", "#A8621F", "#F3E3CF"
DPI = 600


def save(fig, name):
    fig.savefig(name, dpi=DPI, facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.025", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, p, q, lw=1.0, ls="-", color=NAVY, rad=0.0, label=None, lx=0, ly=0, fs=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=fs, ha="center", va="center",
                color=INK, fontweight="bold", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def fig1():
    """Original Figure 1 style: four coloured stage panels with the original icons, rebuilt at full width."""
    import os
    from matplotlib.patches import Rectangle
    fig, ax = canvas(FULL, 3.3)
    here = os.path.dirname(os.path.abspath(__file__))
    icon_dir = os.path.join(os.getcwd(), "..", "fig1")
    stages = [("Traditional\nAccounting", "Manual Ledger & Basic Skills", ["Manual Entries", "Accounting Principles"], "#DCE8F5", "#2F5D8A"),
              ("Computerized\nAccounting", "Excel & ERP Systems", ["Accounting Software", "ERP Systems"], "#CFE0F2", "#2F5D8A"),
              ("Data Analytics\n& AI Integration", "Big Data & Machine Learning", ["Data Analysis", "AI & Predictive Tools"], "#FBE3B5", "#9A5A12"),
              ("Automated &\nIntelligent Accounting", "RPA & Advanced AI", ["Process Automation", "Intelligent Systems"], "#F7CFA4", "#9A5A12")]
    # top arrow: skill & technology advancement
    ax.add_patch(FancyArrowPatch((0.03, 0.935), (0.99, 0.935), arrowstyle="simple,head_width=2.2,head_length=1.4,tail_width=1.3",
                                 mutation_scale=12, fc="#E9A15A", ec="none"))
    ax.text(0.50, 0.935, "Skill & Technology Advancement", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    w, gap, x0, y0, h = 0.232, 0.01, 0.03, 0.115, 0.755
    for i, (t, sub, bullets, fc, tc) in enumerate(stages):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.02", fc=fc, ec="#9FB3C8", lw=0.6))
        ax.text(x + w / 2, y0 + h - 0.075, t, ha="center", va="center", fontsize=8, fontweight="bold", color=tc, linespacing=1.1)
        ax.add_patch(Rectangle((x + 0.008, y0 + h - 0.185), w - 0.016, 0.055, fc="white", ec="none", alpha=0.85))
        ax.text(x + w / 2, y0 + h - 0.158, sub, ha="center", va="center", fontsize=7, color="#222222")
        img = plt.imread(os.path.join(icon_dir, f"icon{i}.png"))
        ih = 0.30
        ax.imshow(img, extent=(x + 0.012, x + w - 0.012, y0 + 0.165, y0 + 0.165 + ih), aspect="auto", zorder=3)
        for k, b in enumerate(bullets):
            ax.text(x + 0.018, y0 + 0.105 - k * 0.06, "•  " + b, ha="left", va="center", fontsize=7.2, color="#111111")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # bottom arrow: educational transformation
    ax.add_patch(FancyArrowPatch((0.03, 0.05), (0.99, 0.05), arrowstyle="simple,head_width=2.2,head_length=1.4,tail_width=1.3",
                                 mutation_scale=12, fc="#4F7DB0", ec="none"))
    ax.text(0.50, 0.05, "Educational Transformation", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    save(fig, "Figure1.png")


def fig2():
    fig, ax = canvas(COL, 4.1)
    steps = [("Research\nobjectives", "Four objectives;\nhypotheses H1–H9"),
             ("Dataset\nconstruction", "Simulated, N = 150;\nanchored on [20]"),
             ("Data\nchecks", "Duplicates, ranges,\nmissing values, outliers"),
             ("Measurement\nmodel", "α, CR, EFA/CFA,\nAVE, HTMT"),
             ("Hypothesis\ntesting", "Hierarchical regression;\nbootstrap mediation"),
             ("Recovery and\nsensitivity", "8 scenarios × 500 replicates;\nfull SEM"),
             ("Interpretation", "Framework implications\nfor teaching")]
    n = len(steps); h, gap, top = 0.115, 0.027, 0.985
    for i, (t, d) in enumerate(steps):
        y = top - h - i * (h + gap)
        box(ax, 0.01, y, 0.37, h, t, fc=LIGHT if i % 2 == 0 else "#DDE6F0", fs=8, bold=True)
        box(ax, 0.43, y, 0.56, h, d, fc="white", ec=MID, fs=7.5)
        arrow(ax, (0.38, y + h / 2), (0.43, y + h / 2), lw=0.7, color=MID)
        if i < n - 1:
            arrow(ax, (0.195, y), (0.195, y - gap))
    save(fig, "Figure2.png")


def fig3():
    fig, ax = canvas(FULL, 3.6)
    ex = [("AI exposure\n(AI_EXP)", 0.80), ("Analytics\nexposure (AN_EXP)", 0.535), ("Automation\nexposure (AU_EXP)", 0.27)]
    out = [("Accounting\nperformance (PERF)", 0.86), ("Critical thinking\n(CT)", 0.655), ("Digital skills\n(DS)", 0.45), ("Student\nsatisfaction (SAT)", 0.245)]
    bw, bh = 0.215, 0.145
    for t, y in ex:
        box(ax, 0.005, y - bh / 2, bw, bh, t, fc=LIGHT, fs=8, bold=True)
    box(ax, 0.40, 0.48, 0.20, 0.15, "Student\nengagement\n(ENG)", fc=ORANGE_F, ec=ORANGE, fs=8, bold=True)
    for t, y in out:
        box(ax, 0.78, y - bh / 2, bw, bh, t, fc="#DDE6F0", fs=8, bold=True)
    X, OX, EL, ER, EY = 0.22, 0.78, 0.40, 0.60, 0.555
    for t, y in ex:
        arrow(ax, (X, y), (EL, EY + (y - 0.535) * 0.18), color=ORANGE, lw=0.9)
    for t, y in out:
        arrow(ax, (ER, EY + (y - 0.55) * 0.18), (OX, y), color=ORANGE, lw=0.9)
    yP, yC, yD, yS = [o[1] for o in out]
    arrow(ax, (X, 0.84), (OX, yP + 0.01), lw=1.1, label="H1", ly=0.02)
    arrow(ax, (X, 0.565), (OX, yC), lw=1.1, rad=-0.25, label="H2", lx=-0.06, ly=0.115)
    arrow(ax, (X, 0.295), (OX, yD), lw=1.1, rad=0.22, label="H3", lx=-0.03, ly=-0.105)
    arrow(ax, (X, 0.235), (OX, yS - 0.02), lw=1.1, ls=(0, (3, 2)), label="H7", lx=0.17, ly=-0.012)
    for x, t in [(0.11, "Technology exposure"), (0.50, "Mediator"), (0.89, "Learning outcomes")]:
        ax.text(x, 0.965, t, ha="center", fontsize=8, color="#444444", style="italic")
    ax.text(0.5, 0.015, "Blue arrows: direct paths H1–H3, and H7 (dashed: all three exposures → satisfaction).\n"
            "Orange arrows: paths through engagement (H4–H6 mediation; H8, H9).  All models include control variables.",
            ha="center", va="bottom", fontsize=7.5, color="#333333", linespacing=1.3)
    save(fig, "Figure3.png")


X = pd.read_excel("reanalysis_results.xlsx", sheet_name=None, index_col=0)
DF = pd.read_csv("simulated_dataset.csv")


def chart(h=2.2, left=0.17, bottom=0.2, right=0.97, top=0.95):
    fig = plt.figure(figsize=(COL, h))
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    ax.grid(axis="y", color="#E3E3E3", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def fig4():
    d = X["descriptives"]; cons = ["AI_EXP", "AN_EXP", "AU_EXP", "ENG", "CT", "DS", "SAT"]
    m, s = [d.loc[c, "mean"] for c in cons], [d.loc[c, "std"] for c in cons]
    fig, ax = chart(h=2.3, bottom=0.24)
    ax.bar(cons, m, color=NAVY, width=0.62, zorder=2)
    ax.errorbar(cons, m, yerr=s, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2.2, zorder=3)
    ax.set_ylim(1, 5); ax.set_ylabel("Mean score (1–5) ± SD")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    save(fig, "Figure4.png")


def std_ci(row, dv="PERF"):
    pre = {"AI_EXP": "AI", "AN_EXP": "AN", "AU_EXP": "AU", "ENG": "ENG"}[row.predictor]
    cols = [c for c in DF if c.startswith(pre) and c[len(pre):].isdigit()]
    f = DF[cols].mean(axis=1).std() / DF[dv].std()
    return row.CI_low * f, row.CI_high * f


def fig5():
    r = X["regression"]; r1 = r[(r.model == "M1") & (~r.predictor.str.startswith("("))]
    labs, b, ps = r1.predictor.tolist(), r1.beta.tolist(), r1.p.tolist()
    ci = [std_ci(row) for _, row in r1.iterrows()]
    fig, ax = chart(h=2.3)
    ax.bar(labs, b, color=[NAVY if p < .05 else "#AFC0D2" for p in ps], width=0.5, zorder=2)
    ax.errorbar(labs, b, yerr=[[v - lo for v, (lo, hi) in zip(b, ci)], [hi - v for v, (lo, hi) in zip(b, ci)]],
                fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2.5, zorder=3)
    ax.axhline(0, color="#333333", lw=0.6)
    for i, (v, (lo, hi), p) in enumerate(zip(b, ci, ps)):
        ax.text(i, hi + 0.02, f"β = {v:.2f}\n{'p < .001' if p < .001 else 'p = ' + format(p, '.3f')[1:]}",
                ha="center", va="bottom", fontsize=7.5, color=INK, linespacing=1.05)
    ax.set_ylabel("Standardized β (95% CI)"); ax.set_ylim(-0.1, 0.9)
    save(fig, "Figure5.png")


def fig6():
    m = X["mediation"]; labs = ["AI_EXP\n→ PERF", "AN_EXP\n→ CT", "AU_EXP\n→ DS"]
    x = np.arange(3); w = 0.37
    fig, ax = chart(h=2.4, bottom=0.2, top=0.86)
    ax.bar(x - w / 2, m.c_total, w, color=NAVY, label="Total (without ENG)", zorder=2)
    ax.bar(x + w / 2, m.c_prime_direct, w, color=LIGHT, edgecolor=NAVY, lw=0.6, hatch="////", label="Direct (with ENG)", zorder=2)
    for i in range(3):
        ax.text(x[i] - w / 2, m.c_total[i] + 0.01, f"{m.c_total[i]:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(x[i] + w / 2, m.c_prime_direct[i] + 0.01, f"{m.c_prime_direct[i]:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, labs); ax.set_ylabel("Standardized effect (β)"); ax.set_ylim(0, 0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handlelength=1.3, columnspacing=0.9)
    save(fig, "Figure6.png")


def fig7():
    m = X["mediation"]; labs = ["AI_EXP → ENG\n→ PERF", "AN_EXP → ENG\n→ CT", "AU_EXP → ENG\n→ DS"]
    y = np.arange(3)[::-1]
    fig = plt.figure(figsize=(COL, 2.2))
    ax = fig.add_axes([0.30, 0.2, 0.67, 0.76])
    ax.grid(axis="x", color="#E3E3E3", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.errorbar(m.indirect_ab, y, xerr=[m.indirect_ab - m.CI95_low, m.CI95_high - m.indirect_ab], fmt="o",
                color=NAVY, ms=4.5, elinewidth=1.0, capsize=2.8, zorder=3)
    ax.axvline(0, color="#333333", lw=0.6, ls="--")
    for yi, v, lo, hi in zip(y, m.indirect_ab, m.CI95_low, m.CI95_high):
        ax.text(v, yi + 0.22, f"{v:.2f} [{lo:.2f}, {hi:.2f}]", ha="center", va="bottom", fontsize=7.5)
    ax.set_yticks(y, labs); ax.set_xlim(-0.02, 0.30); ax.set_ylim(-0.5, 2.75)
    ax.set_xlabel("Indirect effect a×b (95% bootstrap CI)")
    save(fig, "Figure7.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("Figures 1–7 written")
