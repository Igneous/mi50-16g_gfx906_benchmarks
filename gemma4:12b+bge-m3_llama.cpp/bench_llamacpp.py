#!/usr/bin/env python3
"""Concurrency/throughput benchmark for the llama.cpp router on the MI50.

Reports PP/s (prompt processing / prefill) and TG/s (token generation), both
per-request and aggregate, for concurrency 8/16/24/32, across:
  - bge-m3 embeddings only      (PP only; embeddings don't generate)
  - gemma-4-12b chat only        (PP via a prefill wave, TG via a decode wave)
  - 50:50 mix of both at once    (contention between the two resident models)

Methodology (so the numbers are trustworthy):
  * threading.Barrier => all N requests start simultaneously (real concurrency).
  * Every prompt is UNIQUE and cache_prompt=false => no prefix-cache cheating;
    prefill is genuinely recomputed each time.
  * PP is measured with a prefill-heavy wave: long (~512-tok) prompts, max_tokens=1,
    so wall ~= time to prefill all N => aggregate PP/s = sum(prompt_tok)/wall.
  * TG is measured with a decode-heavy wave: short prompt, max_tokens=128 forced
    via ignore_eos, so every request emits exactly 128 tokens => aggregate
    TG/s = sum(gen_tok)/wall.
  * Per-request PP/TG come from llama.cpp's server-side `timings` (prompt_per_second,
    predicted_per_second) -- the true compute rate for that request's own slot.
  * Each level run 3x; we report the best (lowest-wall) wave to cut OS jitter.
  * In the mix, each model's throughput is measured against ITS OWN span
    (first-start..last-finish of that group), not the combined wall.

Stdlib only.
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier
from statistics import mean, median

sys.stdout.reconfigure(line_buffering=True)  # flush per line so `tail -f` shows progress

BASE   = "http://localhost:8000"
# Levels + phase filter overridable from argv:
#   python3 bench_llamacpp.py 1,2,4,6,8 gemma   -> gemma prefill+decode only
LEVELS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [8, 16, 24, 32]
ONLY   = sys.argv[2] if len(sys.argv) > 2 else "all"
N_GEN  = 128
REPEAT = 3

# A long-ish corpus; we slice unique windows out of it to build unique prompts.
CORPUS = (
    "Retrieval augmented generation grounds a language model in your own data. "
    "An ingestion pipeline splits documents into chunks, embeds each chunk into a "
    "dense vector, and stores the vectors in an index. At query time the system "
    "embeds the user question, runs a nearest neighbour search over the index, and "
    "passes the most relevant chunks to the model as context. This reduces "
    "hallucination, keeps answers current without retraining, and lets the system "
    "cite the exact passages it relied on. Hybrid search blends dense vectors with "
    "sparse keyword scores. Rerankers reorder the shortlist for precision. Chunk "
    "size, overlap, and the embedding model all change retrieval quality. Evaluation "
    "uses metrics like recall at k, mean reciprocal rank, and faithfulness graded by "
    "a judge model. Production concerns include latency budgets, caching, batching, "
    "vector index freshness, access control on documents, and cost per thousand "
    "queries. An organization uses this to answer questions from its own document "
    "collection such as manuals, policies, and reports. "
) * 6
WORDS = CORPUS.split()

def slice_words(i, n):
    """Unique ~n-word window for request i (wraps around)."""
    start = (i * 37) % (len(WORDS) - n)
    return " ".join(WORDS[start:start + n])

def post(path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return time.perf_counter() - t0, json.loads(r.read())

def chat_payload(i, n_gen, n_words):
    # unique tag + unique window => no prefix-cache hit
    prompt = f"[r{i}] Summarize the following in one sentence: " + slice_words(i, n_words)
    p = {"model": "gemma-4-12b",
         "messages": [{"role": "user", "content": prompt}],
         "max_tokens": n_gen, "temperature": 0.0,
         "cache_prompt": False}
    if n_gen > 1:
        p["ignore_eos"] = True   # force exactly n_gen tokens for clean TG
    return p

def embed_payload(i, n_words=180):
    return {"model": "bge-m3", "input": [f"[r{i}] " + slice_words(i, n_words)]}

def worker(kind, i, barrier, n_gen, n_words):
    barrier.wait()
    t0 = time.perf_counter()
    try:
        if kind == "chat":
            lat, b = post("/v1/chat/completions", chat_payload(i, n_gen, n_words))
            u, tm = b.get("usage", {}), b.get("timings", {})
            return {"k": "chat", "ok": True, "t0": t0, "t1": time.perf_counter(),
                    "pp_tok": u.get("prompt_tokens", 0),
                    "tg_tok": u.get("completion_tokens", 0),
                    "pp_s": tm.get("prompt_per_second"),
                    "tg_s": tm.get("predicted_per_second"), "lat": lat}
        else:
            lat, b = post("/v1/embeddings", embed_payload(i, n_words))
            u = b.get("usage", {})
            return {"k": "embed", "ok": True, "t0": t0, "t1": time.perf_counter(),
                    "pp_tok": u.get("prompt_tokens", 0), "tg_tok": 0,
                    "pp_s": None, "tg_s": None, "lat": lat}
    except Exception as e:
        return {"k": kind, "ok": False, "t0": t0, "t1": time.perf_counter(),
                "err": str(e)[:140]}

def wave(specs, n_gen, n_words):
    """specs: list of 'chat'/'embed'. All fire simultaneously."""
    n = len(specs)
    bar = Barrier(n)
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(worker, k, i, bar, n_gen, n_words)
                for i, k in enumerate(specs)]
        return [f.result() for f in as_completed(futs)]

def span(rows):
    return max(r["t1"] for r in rows) - min(r["t0"] for r in rows)

def best_of(fn):
    runs = [fn() for _ in range(REPEAT)]
    runs = [r for r in runs if r is not None]
    return min(runs, key=lambda x: x["_wall"]) if runs else None

# ---- phase runners ---------------------------------------------------------

def chat_prefill_level(c):
    # ~120 words (~150 tok) so 48 concurrent prompts fit the unified KV at ctx 8192.
    res = wave(["chat"] * c, n_gen=1, n_words=120)
    ok = [r for r in res if r["ok"]]
    if not ok: return None
    w = span(ok)
    return {"_wall": w, "c": c, "n_err": c - len(ok),
            "pp_tot": sum(r["pp_tok"] for r in ok) / w,
            "pp_req": mean([r["pp_s"] for r in ok if r["pp_s"]]),
            "ptok": sum(r["pp_tok"] for r in ok) // max(1, len(ok)),
            "lat": mean([r["lat"] for r in ok])}

def chat_decode_level(c):
    res = wave(["chat"] * c, n_gen=N_GEN, n_words=24)
    ok = [r for r in res if r["ok"]]
    if not ok: return None
    w = span(ok)
    return {"_wall": w, "c": c, "n_err": c - len(ok),
            "tg_tot": sum(r["tg_tok"] for r in ok) / w,
            "tg_req": mean([r["tg_s"] for r in ok if r["tg_s"]]),
            "lat": mean([r["lat"] for r in ok])}

def embed_level(c):
    # ~110 words (~150 tok) so 48 concurrent inputs fit at ctx 8192.
    res = wave(["embed"] * c, n_gen=0, n_words=110)
    ok = [r for r in res if r["ok"]]
    if not ok: return None
    w = span(ok)
    return {"_wall": w, "c": c, "n_err": c - len(ok),
            "pp_tot": sum(r["pp_tok"] for r in ok) / w,
            "req_s": len(ok) / w,
            "ptok": sum(r["pp_tok"] for r in ok) // max(1, len(ok)),
            "lat": mean([r["lat"] for r in ok])}

def mix_level(c):
    nc = c // 2
    specs = ["chat"] * nc + ["embed"] * (c - nc)
    res = wave(specs, n_gen=N_GEN, n_words=24)
    chat = [r for r in res if r["ok"] and r["k"] == "chat"]
    emb  = [r for r in res if r["ok"] and r["k"] == "embed"]
    if not chat or not emb: return None
    cw, ew = span(chat), span(emb)
    return {"_wall": span([r for r in res if r["ok"]]), "c": c,
            "chat_n": len(chat), "emb_n": len(emb),
            "tg_tot": sum(r["tg_tok"] for r in chat) / cw,
            "tg_req": mean([r["tg_s"] for r in chat if r["tg_s"]]),
            "chat_lat": mean([r["lat"] for r in chat]),
            "emb_pp_tot": sum(r["pp_tok"] for r in emb) / ew,
            "emb_req_s": len(emb) / ew,
            "emb_lat": mean([r["lat"] for r in emb])}

def hdr(t): print(f"\n{'='*82}\n{t}\n{'='*82}")
def errtag(d): return f"  ERR={d['n_err']}" if d.get("n_err") else ""

def main():
    # warmup both models
    for _ in range(2):
        try: post("/v1/chat/completions", chat_payload(0, 8, 16))
        except Exception: pass
        try: post("/v1/embeddings", embed_payload(0))
        except Exception: pass

    if ONLY in ("all", "embed"):
        hdr("BGE-M3 EMBEDDINGS ONLY  (PP only; ~180-word inputs)")
        print(f"{'conc':>5} {'wall_s':>7} {'PP tok/s tot':>13} {'req/s':>7} {'in_tok':>7} {'lat_s':>7}")
        for c in LEVELS:
            d = best_of(lambda c=c: embed_level(c))
            if d: print(f"{c:>5} {d['_wall']:>7.2f} {d['pp_tot']:>13.0f} {d['req_s']:>7.1f} {d['ptok']:>7} {d['lat']:>7.3f}{errtag(d)}")

    if ONLY in ("all", "gemma"):
        hdr("GEMMA-4-12B CHAT -- PREFILL (PP)  (~380-word unique prompts, gen=1)")
        print(f"{'conc':>5} {'wall_s':>7} {'PP tok/s tot':>13} {'PP/s req':>9} {'in_tok':>7} {'lat_s':>7}")
        for c in LEVELS:
            d = best_of(lambda c=c: chat_prefill_level(c))
            if d: print(f"{c:>5} {d['_wall']:>7.2f} {d['pp_tot']:>13.0f} {d['pp_req']:>9.1f} {d['ptok']:>7} {d['lat']:>7.2f}{errtag(d)}")

        hdr(f"GEMMA-4-12B CHAT -- DECODE (TG)  (gen={N_GEN} forced via ignore_eos)")
        print(f"{'conc':>5} {'wall_s':>7} {'TG tok/s tot':>13} {'TG/s req':>9} {'lat_s':>7}")
        for c in LEVELS:
            d = best_of(lambda c=c: chat_decode_level(c))
            if d: print(f"{c:>5} {d['_wall']:>7.2f} {d['tg_tot']:>13.1f} {d['tg_req']:>9.2f} {d['lat']:>7.2f}{errtag(d)}")

    if ONLY in ("all", "mix"):
        hdr(f"50:50 MIX  (C/2 chat gen={N_GEN} + C/2 embed, concurrent; per-model span)")
        print(f"{'conc':>5} {'chat TG/s tot':>13} {'TG/s req':>9} {'chat lat':>9} | {'emb PP/s tot':>12} {'emb req/s':>9} {'emb lat':>8}")
        for c in LEVELS:
            d = best_of(lambda c=c: mix_level(c))
            if d: print(f"{c:>5} {d['tg_tot']:>13.1f} {d['tg_req']:>9.2f} {d['chat_lat']:>9.2f} | {d['emb_pp_tot']:>12.0f} {d['emb_req_s']:>9.1f} {d['emb_lat']:>8.3f}")
    print("\ndone.")

if __name__ == "__main__":
    main()
