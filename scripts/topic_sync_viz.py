"""Per-topic sync/dropout viz for a rosbag, using the KITTI converter's own dt/interp_tol/
backward_interpolation (--pipeline_config) so gaps shown here match what the converter
treats as dropped.
"""
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from tartandriver_utils.ros_utils import stamp_to_time
from tartandriver_utils.os_utils import load_yaml

from ros2bag_2_kitti_multiproc import setup_queue


def compute_sync(bag_dir, topics, dt, backward_interpolation, use_bag_time=False):
    """Nearest-message-per-grid-slot matching, ported from ros2bag_2_kitti_multiproc.py's
    sync pass (not imported, to avoid touching the production converter).

    Returns (queue, msg_times): queue has raw target_times/topic_times/topic_error
    (unfiltered by interp_tol); msg_times is {topic: sorted match times}.
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    bagpath = Path(bag_dir)
    msg_times = {topic: [] for topic in topics}

    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.msgcount > 0 and x.topic in topics]
        queue = setup_queue(reader, list(topics), dt)

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            topic = connection.topic

            if hasattr(msg, "header") and not use_bag_time:
                stamp_time = stamp_to_time(msg.header.stamp)
                msg_time = stamp_time if stamp_time != 0 else timestamp * 1e-9
            else:
                msg_time = timestamp * 1e-9

            msg_times[topic].append(msg_time)

            tdiffs = queue["target_times"] - msg_time
            better_mask = np.abs(tdiffs) < queue["topic_error"][topic]
            if not backward_interpolation:
                better_mask = better_mask & (tdiffs >= 0.)
            queue["topic_error"][topic][better_mask] = np.abs(tdiffs)[better_mask]
            queue["topic_times"][topic][better_mask] = msg_time

    for topic in msg_times:
        msg_times[topic] = np.sort(np.array(msg_times[topic]))

    return queue, msg_times


def _consecutive_spans(times, dt, gap_factor=1.5):
    """Group sorted grid times into [start, end] spans, splitting where the gap
    between consecutive times exceeds gap_factor * dt (i.e. collapse adjacent bad
    grid slots into one annotated dropout span instead of one per slot)."""
    if len(times) == 0:
        return []
    times = np.sort(times)
    spans = []
    start = prev = times[0]
    for t in times[1:]:
        if t - prev > gap_factor * dt:
            spans.append((start, prev + dt))
            start = t
        prev = t
    spans.append((start, prev + dt))
    return spans


def _short_topic(topic):
    """Last two path components of a topic, for compact per-panel y-labels."""
    parts = [p for p in topic.split("/") if p]
    return "/".join(parts[-2:]) if len(parts) > 1 else topic


def _write_sync_csv(target_times, topics, topic_error, interp_tol, out_csv):
    """Per-grid-slot sync data backing the figure: each required/optional topic's
    interpolation error and whether it's within interp_tol, one row per target_time."""
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_time"] +
                         [f"{t}__error_s" for t in topics] +
                         [f"{t}__valid" for t in topics])
        for i, tt in enumerate(target_times):
            errors = [topic_error[t][i] for t in topics]
            valids = [int(e < interp_tol) for e in errors]
            writer.writerow([tt] + errors + valids)
    return out_csv


def render_sync_viz(bag_dir, pipeline_config_path, out_png, out_csv=None, gap_panel=True,
                    use_bag_time=False):
    config = load_yaml(pipeline_config_path)
    dt = config["dt"]
    interp_tol = config["interp_tol"]
    backward_interpolation = config["backward_interpolation"]
    topics = sorted({t["topic"] for t in config["topics"]})
    topic_optional = {t["topic"]: t.get("optional", False) for t in config["topics"]}
    required_topics = [t for t in topics if not topic_optional[t]]

    queue, msg_times = compute_sync(bag_dir, topics, dt, backward_interpolation,
                                     use_bag_time=use_bag_time)
    target_times = queue["target_times"]

    if out_csv is not None:
        _write_sync_csv(target_times, topics, queue["topic_error"], interp_tol, out_csv)

    height_ratios = [0.8] + ([1.4] * len(topics) if gap_panel else [])
    fig, axes = plt.subplots(len(height_ratios), 1, figsize=(14, sum(height_ratios) * 0.9),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": height_ratios})
    axes = axes[:, 0]
    strip_ax = axes[0]
    gap_axes = dict(zip(topics, axes[1:])) if gap_panel else {}

    fig.suptitle(f"Topic sync / dropout (dt={dt}, interp_tol={interp_tol})")

    all_valid = np.ones(len(target_times), dtype=bool)
    for topic in required_topics:
        all_valid &= queue["topic_error"][topic] < interp_tol
    for start, end in _consecutive_spans(target_times[~all_valid], dt):
        strip_ax.axvspan(start, end, color="orange", alpha=0.6)
    strip_ax.set_yticks([])
    strip_ax.set_ylabel("frame\nvalidity", fontsize=7, rotation=0, labelpad=25, va="center")

    for topic, gap_ax in gap_axes.items():
        times = msg_times[topic]
        if len(times) >= 2:
            gap_ax.plot(times[1:], np.diff(times), marker=".", markersize=2,
                        linewidth=0.5, color="tab:blue")
        # dt reference line: gaps above it can't fill every grid slot.
        gap_ax.axhline(dt, color="grey", linestyle="--", linewidth=0.5)

        # Slots where this topic misses interp_tol -- frames it makes the converter
        bad = queue["topic_error"][topic] >= interp_tol
        for start, end in _consecutive_spans(target_times[bad], dt):
            gap_ax.axvspan(start, end, color="red", alpha=0.15)

        label = _short_topic(topic) + ("\n(optional)" if topic_optional[topic] else "")
        gap_ax.set_ylabel(label, fontsize=6, rotation=0, labelpad=32, va="center", ha="right")
        # Log scale: one multi-second dropout otherwise squashes the nominal
        gap_ax.set_yscale("log")
        gap_ax.tick_params(axis="y", labelsize=5)
        gap_ax.grid(True, alpha=0.3, which="both")

    if gap_axes:
        axes[1].set_title("Inter-arrival gap (s) per topic; grey dash = dt, "
                          "red = slots missing interp_tol", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png, out_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag_dir", required=True, help="path to a rosbag run-dir")
    parser.add_argument("--pipeline_config", required=True,
                        help="ros_torch_converter config yaml (same one passed to the KITTI "
                             "converter) -- supplies dt/interp_tol/backward_interpolation/topics")
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--out_csv", default=None,
                        help="output CSV path for per-topic sync data (default: --out with .csv extension)")
    parser.add_argument("--no_gap_panel", action="store_true",
                        help="skip the per-topic inter-arrival gap panels")
    parser.add_argument("--use_bag_time", action="store_true",
                        help="match on bag receive time instead of header.stamp")
    return parser.parse_args()


def main():
    args = parse_args()
    out_csv = args.out_csv or str(Path(args.out).with_suffix(".csv"))
    out_png, out_csv = render_sync_viz(args.bag_dir, args.pipeline_config, args.out,
                                       out_csv=out_csv, gap_panel=not args.no_gap_panel,
                                       use_bag_time=args.use_bag_time)
    print(f"[topic_sync_viz] wrote {out_png}")
    print(f"[topic_sync_viz] wrote {out_csv}")


if __name__ == "__main__":
    main()
