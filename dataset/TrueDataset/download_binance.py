"""
Download the raw monthly 1-second kline archives for the TrueDataset panel.

Source: https://data.binance.vision  (public, no key).  This script throttles
itself to MAX_WORKERS concurrent GETs.

For every (symbol, month) in true_config the script fetches

    {SYMBOL}-1s-{YYYY-MM}.zip
    {SYMBOL}-1s-{YYYY-MM}.zip.CHECKSUM      (one line: "<sha256>  <filename>")

and verifies the SHA256 before accepting the file.  A zip that is already on
disk and checksum-clean is skipped, so the script is resumable: rerun it after
an interruption and it only fetches what is missing.

A month that returns HTTP 404 is reported as MISSING and makes the script exit
non-zero, because a hole in one symbol would silently shift that asset's block
layout relative to the others.

Volume: 8 symbols x 98 months, ~20-70 MB per file, ~24 GB total.

Usage
-----
    python download_binance.py                 # everything
    python download_binance.py --symbols BTCUSDT ETHUSDT
    python download_binance.py --check         # verify what is on disk, fetch nothing
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import true_config as cfg

MAX_WORKERS = 8          # concurrent downloads; the bottleneck is bandwidth, not CPU
CHUNK = 1 << 20          # 1 MiB streaming chunks -- never load a zip into RAM
RETRIES = 4
BACKOFF = 3.0            # seconds, multiplied by attempt index


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_bytes(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _fetch_to_file(url, dest, timeout=600):
    """Stream to a .part file, then atomically rename. Never leaves a truncated zip."""
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)


def expected_sha(symbol, month):
    """The published SHA256, or None if the CHECKSUM file itself is absent."""
    try:
        line = _fetch_bytes(cfg.zip_url(symbol, month) + ".CHECKSUM").decode()
    except urllib.error.HTTPError:
        return None
    return line.split()[0]


def one(symbol, month, check_only=False):
    """Return (symbol, month, status) with status in {ok, cached, missing, error:...}."""
    dest = cfg.zip_path(symbol, month)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    for attempt in range(RETRIES):
        try:
            want = expected_sha(symbol, month)
            if want is None:
                return symbol, month, "missing"

            if os.path.exists(dest) and _sha256(dest) == want:
                return symbol, month, "cached"
            if check_only:
                return symbol, month, "corrupt" if os.path.exists(dest) else "absent"

            _fetch_to_file(cfg.zip_url(symbol, month), dest)
            if _sha256(dest) != want:
                os.remove(dest)
                raise ValueError("sha256 mismatch after download")
            return symbol, month, "ok"

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return symbol, month, "missing"
            if attempt == RETRIES - 1:
                return symbol, month, f"error:HTTP {exc.code}"
        except Exception as exc:                      # noqa: BLE001 - report, then retry
            if attempt == RETRIES - 1:
                return symbol, month, f"error:{type(exc).__name__}: {exc}"
        time.sleep(BACKOFF * (attempt + 1))

    return symbol, month, "error:exhausted"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=cfg.ASSETS)
    ap.add_argument("--start", default=cfg.START_MONTH)
    ap.add_argument("--end", default=cfg.END_MONTH)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--check", action="store_true",
                    help="verify local files against the published checksums, download nothing")
    args = ap.parse_args()

    months = cfg.months(args.start, args.end)
    jobs = [(s, m) for s in args.symbols for m in months]
    print(f"{len(args.symbols)} symbols x {len(months)} months = {len(jobs)} files "
          f"({args.start} .. {args.end}), {args.workers} workers", flush=True)

    counts, problems, done = {}, [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one, s, m, args.check) for s, m in jobs]
        for fut in as_completed(futs):
            symbol, month, status = fut.result()
            done += 1
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key not in ("ok", "cached"):
                problems.append((symbol, month, status))
            if done % 25 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] " +
                      "  ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)

    print("\nsummary: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if problems:
        print(f"\n{len(problems)} problem file(s):")
        for symbol, month, status in sorted(problems):
            print(f"  {symbol:10s} {month}  {status}")
        return 1

    total = sum(os.path.getsize(cfg.zip_path(s, m)) for s, m in jobs)
    print(f"all {len(jobs)} files present and checksum-clean, {total / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
