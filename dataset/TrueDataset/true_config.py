"""
Shared configuration for TrueDataset — a real multi-asset intraday panel built
from Binance public 1-second klines.

Design decisions and the measurements that forced them are recorded in README.md.
The three that constrain every downstream shape:

  * BAR_SECONDS = 30.  Binance emits a complete 86 400-row 1-second grid per day
    per symbol, but fills no-trade seconds with the carried-forward close.  At
    dt = 1s the resulting staleness (45-87% zero returns) collapses the realised
    cross-asset correlation to 0.41 against a 0.71 plateau (Epps effect).  30s is
    the coarsest bar that still admits enough non-overlapping windows to fill the
    benchmark shape, and recovers 0.685 of the 0.708 plateau.

  * SEQ_LEN = 128.  Over 2021-01 .. 2026-07 (2038 days) the 30s panel yields
    45 855 non-overlapping windows of length 128, against the 5 x 8192 = 40 960
    the benchmark contract requires.  Length 252 would yield only 23 306 and
    force overlapping windows.

  * The asset panel is the eight most liquid USDT pairs TODAY, not the eight
    with the longest history.  Measured on three probe months, the long-history
    panel (which is forced to carry TRX and ETC) degrades badly: ETC reaches 72%
    zero 30s-returns and 27% no-trade bars by 2026-06, TRX 53%, and the mean
    pairwise realised correlation of that panel sits at 0.29 in 2018 against
    0.52 later.  The liquid panel instead runs 0.56 (2023-06) to 0.79 (2026-06)
    with no asset above 35% stale.  SOL is the binding listing (2020-09), and
    2021-01 is the binding start because SOL and DOGE are themselves 25% and 40%
    stale during autumn 2020.

Storage contract, inherited from dataset/Heston and dataset/HestonMultiAsset:
arrays store PRICE paths anchored at S0 = 100, NOT log returns.  Any log-return
transform belongs to the model-input layer.
"""
import os

# ---------------------------------------------------------------- the panel
# Eight liquid Binance USDT spot pairs.  First available 1s month per symbol
# (verified against the public bucket):
#   BTC 2017-08  ETH 2017-08  BNB 2017-11  XRP 2018-05
#   ADA 2018-04  LINK 2019-01  DOGE 2019-07  SOL 2020-09
# SOL binds the common start; 2021-01 is chosen instead of 2020-09 because SOL
# and DOGE are still 25% and 40% stale at 30s during autumn 2020.
ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
          "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"]

START_MONTH = "2021-01"
END_MONTH = "2026-07"          # last complete month at time of writing

# ---------------------------------------------------------------- resampling
BAR_SECONDS = 30
SECONDS_PER_DAY = 86_400
assert SECONDS_PER_DAY % BAR_SECONDS == 0, "bar must tile a UTC day exactly"
BARS_PER_DAY = SECONDS_PER_DAY // BAR_SECONDS          # 2880

# Annualised time step.  Crypto trades 24/7, so a year is 365 calendar days.
BARS_PER_YEAR = 365 * BARS_PER_DAY                     # 1_051_200
DT = 1.0 / BARS_PER_YEAR

# ---------------------------------------------------------------- benchmark shape
SEQ_LEN = 128
N_SAMPLES = 8192
S0 = 100.0
D = len(ASSETS)

# Split names and roles match dataset/HestonMultiAsset exactly so the metric
# drivers need no special casing.
SPLITS = ["", "_test", "_disc", "_val", "_valdisc"]

# Interleaved-block layout: BLOCK_WINDOWS contiguous windows per block, blocks
# assigned round-robin to the five splits.  Blocks are spread uniformly over the
# whole history so every split sees every market regime, and the gaps between
# consecutive blocks act as an embargo against boundary autocorrelation.
BLOCK_WINDOWS = 256
assert N_SAMPLES % BLOCK_WINDOWS == 0
BLOCKS_PER_SPLIT = N_SAMPLES // BLOCK_WINDOWS          # 32

# ---------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")                    # monthly 1s kline zips
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

TAG = f"{N_SAMPLES}x{SEQ_LEN}x{D}"


def months(start=START_MONTH, end=END_MONTH):
    """Inclusive list of 'YYYY-MM' strings."""
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def zip_name(symbol, month):
    return f"{symbol}-1s-{month}.zip"


def zip_url(symbol, month):
    return f"{BASE_URL}/{symbol}/1s/{zip_name(symbol, month)}"


def zip_path(symbol, month):
    return os.path.join(RAW_DIR, symbol, zip_name(symbol, month))
