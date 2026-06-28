#!/usr/bin/env python3
"""Extract a single probed layer (default: layer 18) from collected SAFE hidden states.

Each ``policy_records/*.pkl`` written by ``serve_policy.py --record`` holds a
``hidden_states`` array of shape ``(num_steps, num_layers, action_horizon, feature_dim)``,
where the ``num_layers`` axis follows ``SAFE_HIDDEN_LAYERS = (0, 3, 6, 9, 12, 15, 18)``
(so layer 18 is the last entry). This script rewrites each record with ``hidden_states``
reduced to one layer -> ``(num_steps, action_horizon, feature_dim)``, leaving every other
key untouched, into a parallel output directory.

Standalone: depends only on numpy + the stdlib. ``ml_dtypes`` is imported if available so
that bf16-saved records unpickle correctly (it ships with JAX); fp32 records need nothing.

Examples
--------
# slim layer 18 into rollouts/pi0-libero_10/policy_records_layer18/
python scripts/extract_hidden_layer.py rollouts/pi0-libero_10/policy_records

# report sizes without writing anything
python scripts/extract_hidden_layer.py rollouts/pi0-libero_10/policy_records --dry-run

# keep layer 12 instead, also downcast to bf16, and drop the redundant pre_velocity
python scripts/extract_hidden_layer.py rollouts/.../policy_records -l 12 --bf16 --drop-pre-velocity
"""

import argparse
import pathlib
import pickle
import sys

import numpy as np

# Registers the bfloat16 numpy dtype so bf16-saved records can be unpickled. Optional:
# fp32 records (the "original" pre-bf16 data) don't need it.
try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass

# Must match SAFE_HIDDEN_LAYERS in src/openpi/models/pi0.py -- the order of the layers axis.
SAFE_HIDDEN_LAYERS = (0, 3, 6, 9, 12, 15, 18)


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "input_dir", type=pathlib.Path, help="policy_records dir, e.g. rollouts/pi0-libero_10/policy_records"
    )
    ap.add_argument(
        "-o", "--output-dir", type=pathlib.Path, default=None,
        help="where to write slimmed pkls (default: <input_dir>_layer{L})",
    )
    ap.add_argument(
        "-l", "--layer", type=int, default=18,
        help=f"which probed layer to keep; must be in {SAFE_HIDDEN_LAYERS} (default: 18)",
    )
    ap.add_argument(
        "--keepdim", action="store_true",
        help="keep a singleton layer axis -> (num_steps, 1, horizon, feat) instead of squeezing it",
    )
    ap.add_argument("--bf16", action="store_true", help="also downcast the kept layer to bfloat16")
    ap.add_argument(
        "--drop-pre-velocity", action="store_true",
        help="also drop the pre_velocity array (the post-final-norm final feature) if present",
    )
    ap.add_argument(
        "--in-place", action="store_true", help="overwrite the input files instead of writing to a new dir"
    )
    ap.add_argument("--glob", default="*meta.pkl", help="filename pattern to match (default: *meta.pkl)")
    ap.add_argument("-n", "--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    if args.layer not in SAFE_HIDDEN_LAYERS:
        sys.exit(f"error: layer {args.layer} not in SAFE_HIDDEN_LAYERS={SAFE_HIDDEN_LAYERS}")
    layer_idx = SAFE_HIDDEN_LAYERS.index(args.layer)

    in_dir: pathlib.Path = args.input_dir
    if not in_dir.is_dir():
        sys.exit(f"error: not a directory: {in_dir}")

    if args.in_place:
        out_dir = in_dir
    else:
        out_dir = args.output_dir or in_dir.parent / f"{in_dir.name}_layer{args.layer}"
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        sys.exit(f"error: no files matching {args.glob!r} in {in_dir}")

    n_done = n_skip = 0
    bytes_in = bytes_out = 0
    bf16 = None
    if args.bf16:
        import ml_dtypes  # noqa: PLC0415

        bf16 = ml_dtypes.bfloat16

    for f in files:
        with open(f, "rb") as fh:
            rec = pickle.load(fh)

        hs = rec.get("hidden_states")
        if hs is None:
            n_skip += 1
            continue
        hs = np.asarray(hs)
        if hs.ndim != 4 or hs.shape[1] != len(SAFE_HIDDEN_LAYERS):
            print(f"  WARN: unexpected hidden_states shape {hs.shape} in {f.name}; skipping", file=sys.stderr)
            n_skip += 1
            continue

        sel = hs[:, layer_idx : layer_idx + 1] if args.keepdim else hs[:, layer_idx]
        if bf16 is not None:
            sel = sel.astype(bf16)
        rec["hidden_states"] = sel
        rec["hidden_states_layer"] = args.layer  # provenance: which absolute layer was kept

        if args.drop_pre_velocity:
            rec.pop("pre_velocity", None)

        bytes_in += f.stat().st_size
        if not args.dry_run:
            out_path = out_dir / f.name
            with open(out_path, "wb") as fh:
                pickle.dump(rec, fh)
            bytes_out += out_path.stat().st_size
        n_done += 1

    print(f"processed {n_done} files, skipped {n_skip}")
    print(f"input size:  {human(bytes_in)}")
    if not args.dry_run:
        pct = bytes_out / max(bytes_in, 1) * 100
        print(f"output size: {human(bytes_out)}  ({pct:.0f}% of input)")
        print(f"written to:  {out_dir}")
    else:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    main()
