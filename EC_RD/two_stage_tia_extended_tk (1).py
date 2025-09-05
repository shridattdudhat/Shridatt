#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Stage EC Front-End Calculator (Tkinter, Extended v3)
- Separate Stage-1 / Stage-2 inputs & outputs
- Option to "Ignore feed-forward (α=0)"
- Frequency response charts (Stage-1 and Stage-2) using matplotlib if installed
- Total System Gain shown in V/A, |G| in kΩ, and |G| in mV/nA

DC formulas:
  Stage 1 (TIA):  G1_dc = -Rf1 (V/A),      fc1 = 1/(2π Rf1 Cf1)
  Stage 2 (eff):  β = Rf2/Rin2,  α = R21B/(R21B + R23)  (unless "ignore" is checked → α=0)
                  Zf(s) = Rf2 / (1 + s Rf2 Cf2)
                  G2_eff(s) = (1 + Zf/Rin2)*α - (Zf/Rin2)
                  For DC, Zf → Rf2 ⇒ G2_eff = (1+β)α - β
  Total I→V:      G_total = G1_dc · G2_eff(DC)

Run on Windows:
  python two_stage_tia_extended_tk.py
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Attempt to import matplotlib for plotting; fail gracefully if not installed
HAS_MPL = True
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    HAS_MPL = False

def corner_freq(r, c):
    if r <= 0 or c <= 0:
        return float("inf")
    return 1.0 / (2.0 * math.pi * r * c)

def fmt_si(x, unit=""):
    if x is None or math.isinf(x):
        return f"∞ {unit}".strip()
    prefixes = [
        (1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"),
        (1, ""), (1e3, "k"), (1e6, "M"), (1e9, "G")
    ]
    mag = abs(x) if x != 0 else 1.0
    best = min(prefixes, key=lambda t: abs(math.log10(mag/t[0])))
    return f"{x/best[0]:.4g} {best[1]}{unit}".strip()

def parse_float(entry: tk.Entry, name: str) -> float:
    raw = entry.get().strip()
    try:
        return float(raw)
    except Exception:
        raise ValueError(f"Invalid value for {name}: {raw}")

def safe_set(label: tk.Label, text: str):
    label.config(text=text)

def clear_outputs():
    for lab in (v_g1_va, v_g1_kohm, v_fc1_hz,
                v_beta, v_alpha, v_g2eff, v_fc2_hz,
                v_gt_va, v_gt_kohm, v_gt_mv_per_na, v_alpha_tip):
        safe_set(lab, "—")
    if HAS_MPL:
        for canv in (canvas1_holder.get("canvas"), canvas2_holder.get("canvas")):
            if canv is not None:
                canv.get_tk_widget().destroy()
        canvas1_holder["canvas"] = None
        canvas2_holder["canvas"] = None

def compute():
    clear_outputs()
    try:
        # Stage-1 inputs
        rf1  = parse_float(e_rf1,  "Rf1 (Ω)")
        cf1  = parse_float(e_cf1,  "Cf1 (F)")

        # Stage-2 inputs
        rin2 = parse_float(e_rin2, "Rin2 (Ω)")
        rf2  = parse_float(e_rf2,  "Rf2 (Ω)")
        cf2  = parse_float(e_cf2,  "Cf2 (F)")

        # Feed-forward
        r21b = parse_float(e_r21b, "R21B (Ω)")
        r23  = parse_float(e_r23,  "R23 (Ω)")

        ignore_ff = var_ignore_ff.get()

        # ---- Stage 1 ----
        g1_dc = -rf1                          # V/A
        fc1   = corner_freq(rf1, cf1)

        # ---- Stage 2 ----
        if rin2 == 0:
            raise ValueError("Rin2 must be non-zero")
        beta = rf2 / rin2
        if ignore_ff:
            alpha = 0.0
        else:
            if r21b <= 0 and r23 <= 0:
                alpha = 0.0
            elif r21b <= 0:
                alpha = 0.0
            elif r23 <= 0:
                alpha = 1.0
            else:
                alpha = r21b / (r21b + r23)

        # DC effective gain
        g2_eff_dc = (1.0 + beta) * alpha - beta  # V/V
        fc2       = corner_freq(rf2, cf2)

        # ---- Total ----
        g_total = g1_dc * g2_eff_dc            # V/A
        g_total_mag = abs(g_total)             # V/A
        g1_kohm = abs(g1_dc) / 1e3
        gt_kohm = g_total_mag / 1e3
        gt_mv_per_na = g_total_mag * 1e-3      # mV/nA

        # Update Stage-1 outputs
        safe_set(v_g1_va,    f"{g1_dc:.3e} V/A  ({fmt_si(g1_dc,'Ω')})")
        safe_set(v_g1_kohm,  f"{g1_kohm:.3f} kΩ")
        safe_set(v_fc1_hz,   f"{fc1:.4g} Hz  ({fmt_si(fc1,'Hz')})")

        # Update Stage-2 outputs
        safe_set(v_beta,     f"{beta:.6g}")
        safe_set(v_alpha,    f"{alpha:.6g}" + ("  (ignored)" if ignore_ff else ""))
        safe_set(v_g2eff,    f"{g2_eff_dc:.6g} V/V")
        safe_set(v_fc2_hz,   f"{fc2:.4g} Hz  ({fmt_si(fc2,'Hz')})")

        # Update Total
        safe_set(v_gt_va,        f"{g_total:.3e} V/A")
        safe_set(v_gt_kohm,      f"{gt_kohm:.3f} kΩ  (|G|)")
        safe_set(v_gt_mv_per_na, f"{gt_mv_per_na:.3f} mV/nA")

        # Tip for alpha cancellation
        alpha_cancel = beta / (1.0 + beta) if (1.0 + beta) != 0 else float("inf")
        safe_set(v_alpha_tip, f"α_cancel ≈ {alpha_cancel:.6g}  (to suppress DC feedthrough)")

        # ---- Plots ----
        if HAS_MPL:
            # Frequency axis
            fmin, fmax = 0.5, 2000.0
            f = np.logspace(np.log10(fmin), np.log10(fmax), 800)
            w = 2*np.pi*f
            s = 1j*w

            # Stage-1 transfer (V/I): -Rf1 / (1 + s Rf1 Cf1)
            Zf1 = rf1 / (1 + s*rf1*cf1)
            G1s = -Zf1  # V/I
            mag1 = 20*np.log10(np.abs(G1s))

            fig1 = plt.figure(figsize=(5,3))
            ax1 = fig1.add_subplot(111)
            ax1.semilogx(f, mag1)
            ax1.set_xlabel("Frequency [Hz]")
            ax1.set_ylabel("Gain [dB] (V/I)")
            ax1.set_title("Stage 1 (TIA) — Magnitude")
            ax1.grid(True, which="both", linestyle=":")

            canvas1 = FigureCanvasTkAgg(fig1, master=sec_plot1)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill="both", expand=True)
            canvas1_holder["canvas"] = canvas1

            # Stage-2 transfer (V/V): G2_eff(s) = (1 + Zf/Rin2)*α - (Zf/Rin2)
            Zf2 = rf2 / (1 + s*rf2*cf2)
            G2s = (1 + Zf2/rin2)*alpha - (Zf2/rin2)
            mag2 = 20*np.log10(np.abs(G2s))

            fig2 = plt.figure(figsize=(5,3))
            ax2 = fig2.add_subplot(111)
            ax2.semilogx(f, mag2)
            ax2.set_xlabel("Frequency [Hz]")
            ax2.set_ylabel("Gain [dB] (V/V)")
            ax2.set_title("Stage 2 (Effective) — Magnitude")
            ax2.grid(True, which="both", linestyle=":")

            canvas2 = FigureCanvasTkAgg(fig2, master=sec_plot2)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)
            canvas2_holder["canvas"] = canvas2
        else:
            safe_set(lbl_plot_hint, "Plots require matplotlib. Install with: pip install matplotlib")

    except Exception as ex:
        messagebox.showerror("Error", str(ex))

def save_csv():
    try:
        rf1  = parse_float(e_rf1,  "Rf1 (Ω)")
        cf1  = parse_float(e_cf1,  "Cf1 (F)")
        rin2 = parse_float(e_rin2, "Rin2 (Ω)")
        rf2  = parse_float(e_rf2,  "Rf2 (Ω)")
        cf2  = parse_float(e_cf2,  "Cf2 (F)")
        r21b = parse_float(e_r21b, "R21B (Ω)")
        r23  = parse_float(e_r23,  "R23 (Ω)")
        ignore_ff = var_ignore_ff.get()

        g1_dc = -rf1
        fc1   = corner_freq(rf1, cf1)
        if rin2 == 0:
            raise ValueError("Rin2 must be non-zero")
        beta = rf2 / rin2
        if ignore_ff:
            alpha = 0.0
        else:
            alpha = 0.0 if r21b <= 0 else (1.0 if r23 <= 0 else r21b/(r21b + r23))
        g2_eff_dc = (1.0 + beta) * alpha - beta
        fc2    = corner_freq(rf2, cf2)
        g_total = g1_dc * g2_eff_dc
        gt_kohm = abs(g_total)/1e3
        gt_mv_per_na = abs(g_total) * 1e-3
        g1_kohm = abs(g1_dc)/1e3

        rows = [
            ["SECTION","PARAMETER","VALUE RAW","PRETTY/NOTES"],
            ["Stage-1 INPUT","Rf1 (Ω)", rf1, fmt_si(rf1,"Ω")],
            ["Stage-1 INPUT","Cf1 (F)", cf1, fmt_si(cf1,"F")],
            ["Stage-2 INPUT","Rin2 (Ω)", rin2, fmt_si(rin2,"Ω")],
            ["Stage-2 INPUT","Rf2 (Ω)", rf2, fmt_si(rf2,"Ω")],
            ["Stage-2 INPUT","Cf2 (F)", cf2, fmt_si(cf2,"F")],
            ["Feed-Forward INPUT","R21B (Ω)", r21b, fmt_si(r21b,"Ω")],
            ["Feed-Forward INPUT","R23 (Ω)", r23, fmt_si(r23,"Ω")],
            ["OPTION","Ignore feed-forward (α=0)?", ignore_ff, ""],
            ["Stage-1 OUTPUT","G1_dc (V/A)", g1_dc, fmt_si(g1_dc,"Ω")],
            ["Stage-1 OUTPUT","G1_dc (kΩ)", g1_kohm, "kΩ"],
            ["Stage-1 OUTPUT","fc1 (Hz)", fc1, fmt_si(fc1,"Hz")],
            ["Stage-2 OUTPUT","β = Rf2/Rin2", beta, ""],
            ["Stage-2 OUTPUT","α", alpha, ""],
            ["Stage-2 OUTPUT","G2_eff (DC, V/V)", g2_eff_dc, ""],
            ["Stage-2 OUTPUT","fc2 (Hz)", fc2, fmt_si(fc2,"Hz")],
            ["TOTAL OUTPUT","G_total (V/A)", g_total, ""],
            ["TOTAL OUTPUT","|G_total| (kΩ)", gt_kohm, "kΩ"],
            ["TOTAL OUTPUT","|G_total| (mV/nA)", gt_mv_per_na, "mV/nA"],
        ]

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")],
            title="Save results as CSV"
        )
        if not path:
            return
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        messagebox.showinfo("Saved", f"Saved to:\n{path}")
    except Exception as ex:
        messagebox.showerror("Error", str(ex))

# ---- UI ----
root = tk.Tk()
root.title("Two-Stage EC Front-End Calculator (Extended v3, Tkinter)")
root.geometry("1100x780")

pad = {'padx': 10, 'pady': 6}
frm = ttk.Frame(root)
frm.pack(fill="both", expand=True, **pad)

# ===== Inputs: Stage 1 =====
sec_in1 = ttk.LabelFrame(frm, text="Inputs — Stage 1 (TIA)")
sec_in1.grid(row=0, column=0, sticky="nsew", **pad)

def add_in(parent, label, default, row):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
    ent = ttk.Entry(parent, width=20)
    ent.insert(0, str(default))
    ent.grid(row=row, column=1, sticky="w")
    return ent

e_rf1  = add_in(sec_in1, "Rf1 (Ω)", 147e3, 0)
e_cf1  = add_in(sec_in1, "Cf1 (F)", 220e-9, 1)

# ===== Inputs: Stage 2 =====
sec_in2 = ttk.LabelFrame(frm, text="Inputs — Stage 2 (Inverting + LPF)")
sec_in2.grid(row=0, column=1, sticky="nsew", **pad)

e_rin2 = add_in(sec_in2, "Rin2 (Ω)", 180e3, 0)
e_rf2  = add_in(sec_in2, "Rf2 (Ω)", 1e6,   1)
e_cf2  = add_in(sec_in2, "Cf2 (F)", 10e-9, 2)

# ===== Inputs: Feed-forward =====
sec_in3 = ttk.LabelFrame(frm, text="Inputs — Feed-Forward to + input")
sec_in3.grid(row=0, column=2, sticky="nsew", **pad)

e_r21b = add_in(sec_in3, "R21B (Ω) [+ input from Vin]", 200e3, 0)
e_r23  = add_in(sec_in3, "R23 (Ω)  [+ input to GND]",   1e6,   1)

# Option
sec_opt = ttk.LabelFrame(frm, text="Options")
sec_opt.grid(row=1, column=0, columnspan=3, sticky="nsew", **pad)
var_ignore_ff = tk.BooleanVar(value=False)
ttk.Checkbutton(sec_opt, text="Ignore feed-forward (set α = 0)", variable=var_ignore_ff).grid(row=0, column=0, sticky="w")

# Buttons
btns = ttk.Frame(frm)
btns.grid(row=2, column=0, columnspan=3, sticky="w", **pad)
ttk.Button(btns, text="Compute / Plot", command=compute).grid(row=0, column=0, **pad)
ttk.Button(btns, text="Save CSV…", command=save_csv).grid(row=0, column=1, **pad)
ttk.Button(btns, text="Quit", command=root.destroy).grid(row=0, column=2, **pad)

# ===== Outputs: Stage 1 =====
sec_out1 = ttk.LabelFrame(frm, text="Outputs — Stage 1")
sec_out1.grid(row=3, column=0, sticky="nsew", **pad)

def add_out(parent, label, row):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
    lab = ttk.Label(parent, text="—", font=("Segoe UI", 10, "bold"))
    lab.grid(row=row, column=1, sticky="w")
    return lab

v_g1_va   = add_out(sec_out1, "Stage-1 DC gain (V/A):", 0)
v_g1_kohm = add_out(sec_out1, "Stage-1 DC gain (kΩ):", 1)
v_fc1_hz  = add_out(sec_out1, "Stage-1 corner f_c1 (Hz):", 2)

# ===== Outputs: Stage 2 =====
sec_out2 = ttk.LabelFrame(frm, text="Outputs — Stage 2")
sec_out2.grid(row=3, column=1, sticky="nsew", **pad)

v_beta   = add_out(sec_out2, "β = Rf2/Rin2:", 0)
v_alpha  = add_out(sec_out2, "α (feed-forward):", 1)
v_g2eff  = add_out(sec_out2, "Stage-2 effective DC gain (V/V):", 2)
v_fc2_hz = add_out(sec_out2, "Stage-2 corner f_c2 (Hz):", 3)

# ===== Outputs: Total =====
sec_out3 = ttk.LabelFrame(frm, text="Outputs — Total System Gain")
sec_out3.grid(row=3, column=2, sticky="nsew", **pad)

v_gt_va        = add_out(sec_out3, "Total System Gain (I→V, V/A):", 0)
v_gt_kohm      = add_out(sec_out3, "Total System Gain (|G| in kΩ):", 1)
v_gt_mv_per_na = add_out(sec_out3, "Total System Gain (|G| in mV/nA):", 2)
v_alpha_tip    = add_out(sec_out3, "Tip:", 3)

# ===== Plots =====
sec_plot_parent = ttk.LabelFrame(frm, text="Frequency Response (Magnitude)")
sec_plot_parent.grid(row=4, column=0, columnspan=3, sticky="nsew", **pad)

sec_plot1 = ttk.LabelFrame(sec_plot_parent, text="Stage 1 — |Gain| in dB (V/I)")
sec_plot1.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
sec_plot2 = ttk.LabelFrame(sec_plot_parent, text="Stage 2 — |Gain| in dB (V/V)")
sec_plot2.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

canvas1_holder = {"canvas": None}
canvas2_holder = {"canvas": None}

lbl_plot_hint = ttk.Label(sec_plot_parent, text="" if HAS_MPL else "Plots require matplotlib. Install with: pip install matplotlib")
lbl_plot_hint.grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=6)

# First compute with defaults so fields populate
compute()

root.mainloop()
