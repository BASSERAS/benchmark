"""
COSCI-GAN paper reproduction — EEG Table 4 correlation-matrix MAE.

Faithful port of the upstream evaluation
  Experiments/Correlation_Analysis/EEG_Dataset/Time_Series_Analysis_EEG.ipynb

Metric (Table 4 of Seyfi et al., NeurIPS 2022):
  1. For each of the 5 EEG channels, compute the 22 catch22 features on every
     one of the 1024 windows  ->  a (1024, 22) feature frame per channel.
  2. For every channel pair (i, j) with j >= i (15 pairs, diagonal included)
     build the cross-channel feature-correlation block (channel-i features vs
     channel-j features), then drop the two DFA/fluctuation features that are
     degenerate (constant -> NaN correlation), leaving a 20x20 block.
  3. MAE(pair) = mean |corr_real - corr_gen| over the 20x20 block.
  4. Report mean and std of MAE across the 15 pairs (matches the notebook's
     np.mean/np.std over MAEs_*).

Both real and generated series are laid out exactly as upstream: the CSV rows
are block-channel-major (channel 0's 100 samples, then channel 1's, ...), and
BOTH real and generated arrays are produced with reshape(N, -1, 5) so they are
mutually consistent (this is what makes the absolute 0.111 meaningful).

Usage:
  python eeg_corr_mae.py --real <real_5best_label.csv> --gen <generated.npy> [--tag NAME]
  # validation mode: pass the upstream npy files to reproduce Table 4
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from pycatch22 import catch22_all

# The two features upstream drops (degenerate / NaN correlations on EEG windows)
DROP_FEATURES = [
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
]


def catch22_frame(arr_2d):
    """arr_2d: (N, T) -> DataFrame (N, 22) of catch22 features."""
    names = catch22_all(arr_2d[0])["names"]
    feats = np.zeros((arr_2d.shape[0], 22))
    for k, row in enumerate(arr_2d):
        feats[k] = catch22_all(row)["values"]
    return pd.DataFrame(feats, columns=names)


def build_channel_frames(data_3d, prefix):
    """data_3d: (N, T, C) -> dict{c: (N,22) frame prefixed 'prefix_c_'}."""
    C = data_3d.shape[-1]
    out = {}
    for c in range(C):
        out[c] = catch22_frame(data_3d[:, :, c]).add_prefix(f"{prefix}_{c}_")
    return out


def corr_mae(df_real, df_gen, n_channels, prefix_real, prefix_gen):
    """Mean/std of cross-channel feature-correlation MAE over all channel pairs."""
    maes = []
    for i in range(n_channels):
        for j in range(i, n_channels):
            real = pd.concat([df_real[i], df_real[j]], axis=1).corr()
            real = real[list(df_real[j].columns)].loc[list(df_real[i].columns)]
            gen = pd.concat([df_gen[i], df_gen[j]], axis=1).corr()
            gen = gen[list(df_gen[j].columns)].loc[list(df_gen[i].columns)]

            rj = [f"{prefix_real}_{j}_{f}" for f in DROP_FEATURES]
            ri = [f"{prefix_real}_{i}_{f}" for f in DROP_FEATURES]
            gj = [f"{prefix_gen}_{j}_{f}" for f in DROP_FEATURES]
            gi = [f"{prefix_gen}_{i}_{f}" for f in DROP_FEATURES]

            real_b = np.array(real.drop(rj, axis=1).drop(ri, axis=0))
            gen_b = np.array(gen.drop(gj, axis=1).drop(gi, axis=0))
            maes.append(float(np.mean(np.abs(real_b - gen_b))))
    return float(np.mean(maes)), float(np.std(maes)), maes


def load_real(csv_path, n_channels):
    r = np.array(pd.read_csv(csv_path))
    return r.reshape(r.shape[0], -1, n_channels)  # (N, T, C), upstream convention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--gen", required=True, help=".npy (N,T,C) or block CSV")
    ap.add_argument("--n_channels", type=int, default=5)
    ap.add_argument("--tag", default="gen")
    args = ap.parse_args()

    real_data = load_real(args.real, args.n_channels)
    if args.gen.endswith(".npy"):
        gen_data = np.load(args.gen)
    else:
        g = np.array(pd.read_csv(args.gen))
        gen_data = g.reshape(g.shape[0], -1, args.n_channels)

    assert gen_data.shape[-1] == args.n_channels, gen_data.shape
    df_real = build_channel_frames(real_data, "real")
    df_gen = build_channel_frames(gen_data, args.tag)
    mean, std, maes = corr_mae(df_real, df_gen, args.n_channels, "real", args.tag)
    print(f"[{args.tag}] corr-MAE mean={mean:.4f} std={std:.4f} (over {len(maes)} pairs)")
    print("per-pair:", [round(m, 4) for m in maes])


if __name__ == "__main__":
    main()
