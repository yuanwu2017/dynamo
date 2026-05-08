#!/usr/bin/env python3
# Simple async benchmark for dynamo OpenAI endpoint.
# Measures TTFT, total latency, and throughput at fixed concurrency.
import argparse
import asyncio
import json
import statistics
import time

import aiohttp


PROMPT_PREFIX = (
    "You are a careful, concise assistant. "
    "Repeat the following passage verbatim, word for word, "
    "without any additions, then continue writing.\n\n"
)
# A ~100-token chunk of filler we can repeat to scale ISL.
FILLER = (
    "The quick brown fox jumps over the lazy dog near the river bank. "
    "Birds sing in the morning while gentle wind moves through tall trees. "
    "Children play in the park as sunlight filters through green leaves. "
    "A small boat drifts on calm water carrying baskets of fresh apples. "
)


def build_prompt(target_tokens: int) -> str:
    # ~7 chars/token approximation; add filler until we hit target_tokens.
    body = FILLER * max(1, target_tokens // 18)
    return PROMPT_PREFIX + body


async def one_request(session, url, model, prompt, max_tokens, idx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    ttft = None
    n_completion = 0
    err = None
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status != 200:
                err = f"HTTP {resp.status}: {(await resp.text())[:200]}"
            else:
                async for raw in resp.content:
                    if not raw:
                        continue
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:"):].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        n_completion += 1
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    total = time.perf_counter() - t0
    return {"idx": idx, "ttft": ttft, "total": total, "n": n_completion, "err": err}


async def run(args):
    prompt = build_prompt(args.isl)
    url = f"{args.url}/v1/chat/completions"
    sem = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession() as session:
        async def gated(i):
            async with sem:
                return await one_request(session, url, args.model, prompt, args.osl, i)

        # warmup
        for _ in range(args.warmup):
            await gated(-1)

        t_start = time.perf_counter()
        results = await asyncio.gather(*[gated(i) for i in range(args.requests)])
        t_end = time.perf_counter()

    ok = [r for r in results if r["err"] is None and r["ttft"] is not None]
    fail = [r for r in results if r["err"] is not None]
    if not ok:
        print("ALL FAILED")
        for r in fail[:3]:
            print(r)
        return

    ttfts = sorted(r["ttft"] for r in ok)
    totals = sorted(r["total"] for r in ok)
    completions = sum(r["n"] for r in ok)

    def pct(values, p):
        idx = max(0, int(len(values) * p / 100) - 1)
        return values[idx]

    elapsed = t_end - t_start
    out = {
        "label": args.label,
        "isl_target": args.isl,
        "osl": args.osl,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "ok": len(ok),
        "failed": len(fail),
        "elapsed_s": round(elapsed, 3),
        "throughput_req_s": round(len(ok) / elapsed, 3),
        "throughput_completion_tok_s": round(completions / elapsed, 1),
        "ttft_avg_ms": round(1000 * statistics.mean(ttfts), 2),
        "ttft_p50_ms": round(1000 * pct(ttfts, 50), 2),
        "ttft_p90_ms": round(1000 * pct(ttfts, 90), 2),
        "ttft_p99_ms": round(1000 * pct(ttfts, 99), 2),
        "total_avg_ms": round(1000 * statistics.mean(totals), 2),
        "total_p50_ms": round(1000 * pct(totals, 50), 2),
    }
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--isl", type=int, default=1024, help="approx input tokens (controls KV size)")
    p.add_argument("--osl", type=int, default=64, help="max output tokens")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--label", default="run")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
