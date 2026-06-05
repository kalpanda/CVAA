import pandas as pd
import numpy as np


def compute_ad_fd(ref_npz_path, var_npz_path):
    """
    Compute AD (Average Deviation) and FD (Final Deviation) between two
    trajectory distributions, using the mean-over-candidates approach from
    analyse_deviation_v4.

    Both the reference and variant npz files are summarised by their mean
    trajectory (mean over K candidates at each timestep). AD and FD then
    measure the displacement between these two distribution means.

    This is seed-stable: the mean over K candidates is much less sensitive
    to which particular candidates were drawn than any single candidate or
    min-over-candidates. Rank stability across seeds is therefore a property
    of the object's genuine causal influence, not lucky draws.

    AD = mean over timesteps of ||mean_variant_traj - mean_ref_traj||
    FD = ||mean_variant_traj[-1] - mean_ref_traj[-1]||

    npz schema:
        pred_xyz : [num_traj, T, 3]

    Returns (AD, FD) as floats, or (None, None) on load failure.
    """
    try:
        ref_data = np.load(ref_npz_path)
        var_data = np.load(var_npz_path)
    except Exception:
        return None, None

    ref_pred = ref_data["pred_xyz"]   # [K,  T, 3]
    var_pred = var_data["pred_xyz"]   # [K', T, 3]

    # Mean trajectory over all candidates (XY only)
    ref_mean = ref_pred[:, :, :2].mean(axis=0)   # [T, 2]
    var_mean = var_pred[:, :, :2].mean(axis=0)   # [T, 2]

    # Per-timestep displacement between the two distribution means
    disp = np.linalg.norm(var_mean - ref_mean, axis=-1)  # [T]

    AD = float(disp.mean())   # mean over all timesteps
    FD = float(disp[-1])      # final timestep only
    return AD, FD


def main(csv_path, output_path,
         group_col="scene_name",
         ref_col="is_reference",
         traj_col="traj_npz"):

    df = pd.read_csv(csv_path)

    ad_values = []
    fd_values = []

    for group_id, group_df in df.groupby(group_col):

        ref_rows = group_df[group_df[ref_col] == True]
        if ref_rows.empty:
            # No reference trajectory for this group — skip
            ad_values.extend([np.nan] * len(group_df))
            fd_values.extend([np.nan] * len(group_df))
            continue

        ref_npz_path = ref_rows.iloc[0][traj_col]

        for _, row in group_df.iterrows():
            if row[ref_col]:
                # Reference row: deviation from itself is zero by definition
                ad_values.append(0.0)
                fd_values.append(0.0)
                continue

            ad, fd = compute_ad_fd(ref_npz_path, row[traj_col])
            ad_values.append(ad if ad is not None else np.nan)
            fd_values.append(fd if fd is not None else np.nan)

    df["AD"] = ad_values
    df["FD"] = fd_values

    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("output_path")
    p.add_argument("--group_col", default="scene_name")
    p.add_argument("--ref_col",   default="is_reference")
    p.add_argument("--traj_col",  default="traj_npz")
    args = p.parse_args()
    main(args.csv_path, args.output_path,
         args.group_col, args.ref_col, args.traj_col)