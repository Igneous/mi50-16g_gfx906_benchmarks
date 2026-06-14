#!/usr/bin/env python3
"""Generate SVG charts for the gfx906 llama.cpp benchmark report.
Run with the venv that has matplotlib:  /tmp/plotenv/bin/python report/plot.py
Data is the consolidated benchmark matrix (see bench-*.log). Config: b9623,
gemma-4-12b Q4_K_M f16-KV + bge-m3 FP16, ctx 8192, parallel 48."""
import os
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
CONC = [1, 2, 4, 6, 8, 16, 32, 48]

# colors/styles per backend
ST = {
    "ROCm 7.2.3": dict(color="#1f6feb", marker="o", ls="-",  lw=2),
    "ROCm 6.3.3": dict(color="#2da44e", marker="s", ls="--", lw=1.6),
    "Vulkan (RADV)": dict(color="#fb8500", marker="^", ls="-", lw=2),
}

DECODE_TOTAL = {
    "ROCm 7.2.3":   [43.0, 57.6, 67.8, 69.0, 73.2, 157.3, 157.8, 186.5],
    "ROCm 6.3.3":   [40.4, 54.3, 57.3, 61.7, 64.2, 145.0, 102.6, 201.2],
    "Vulkan (RADV)":[39.9, 61.6, 47.9, 68.7, 90.1,  82.9, 139.3, 159.1],
}
DECODE_REQ = {
    "ROCm 7.2.3":   [47.1, 31.5, 18.5, 12.5, 9.96, 11.6, 5.8, 4.8],
    "ROCm 6.3.3":   [43.7, 29.4, 15.5, 11.1, 8.67, 11.2, 4.7, 5.4],
    "Vulkan (RADV)":[43.7, 34.2, 13.1, 12.7, 12.6,  5.8, 5.2, 4.1],
}
PREFILL_TOTAL = {
    "ROCm 7.2.3":   [422, 448, 479, 488, 479, 494, 496, 492],
    "ROCm 6.3.3":   [409, 432, 472, 473, 469, 487, 467, 483],
    "Vulkan (RADV)":[357, 376, 369, 404, 425, 402, 388, 413],
}
EMBED_TOTAL = {
    "ROCm 7.2.3":   [5514, 5677, 9557, 9590, 9667, 9726, 9606, 9498],
    "ROCm 6.3.3":   [5501, 5714, 9487, 9733, 9304, 9479, 9240, 9665],
    "Vulkan (RADV)":[2773, 4340, 4335, 4342, 4370, 4362, 4277, 4315],
}

def line_chart(data, title, ylabel, fname, ymin=0):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, ys in data.items():
        ax.plot(CONC, ys, label=name, markersize=5, **ST[name])
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONC); ax.set_xticklabels([str(c) for c in CONC])
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(bottom=ymin)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)

line_chart(DECODE_TOTAL, "Gemma-4-12B decode throughput (aggregate)",
           "tokens/sec (all requests)", "decode-total.svg")
line_chart(DECODE_REQ, "Gemma-4-12B decode throughput (per request)",
           "tokens/sec per request", "decode-perreq.svg")
line_chart(PREFILL_TOTAL, "Gemma-4-12B prefill throughput (aggregate)",
           "tokens/sec (all requests)", "prefill-total.svg")
line_chart(EMBED_TOTAL, "BGE-M3 embedding throughput (aggregate)",
           "prompt tokens/sec (all requests)", "embed-total.svg")

# --- KV cache dtype demo (ROCm 7.2.3, conc=4) ---
KV = ["f16", "q8_0", "q4_0"]
KV_TGREQ = [18.45, 17.95, 16.88]
KV_VRAM  = [13.41, 11.44, 10.45]
fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
x = range(len(KV)); w = 0.38
b1 = ax1.bar([i - w/2 for i in x], KV_TGREQ, w, color="#1f6feb", label="decode tok/s/req")
ax1.set_ylabel("decode tokens/sec per request", color="#1f6feb")
ax1.tick_params(axis="y", labelcolor="#1f6feb")
ax1.set_ylim(0, 22)
ax2 = ax1.twinx()
b2 = ax2.bar([i + w/2 for i in x], KV_VRAM, w, color="#bc8f00", label="VRAM used (GB)")
ax2.set_ylabel("VRAM used (GB)  /16", color="#bc8f00")
ax2.tick_params(axis="y", labelcolor="#bc8f00")
ax2.set_ylim(0, 16); ax2.axhline(16, color="#cc0000", ls=":", lw=1)
ax1.set_xticks(list(x)); ax1.set_xticklabels(KV)
ax1.set_xlabel("KV cache dtype")
ax1.set_title("KV cache dtype: speed vs VRAM (gemma-4-12b, conc=4, ROCm 7.2.3)",
              fontsize=10.5, fontweight="bold")
for i, v in enumerate(KV_TGREQ): ax1.text(i - w/2, v + 0.2, f"{v:.1f}", ha="center", fontsize=8, color="#1f6feb")
for i, v in enumerate(KV_VRAM):  ax2.text(i + w/2, v + 0.2, f"{v:.1f}", ha="center", fontsize=8, color="#7a5c00")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "kvcache.svg"))
plt.close(fig)
print("wrote kvcache.svg")
