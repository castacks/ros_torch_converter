"""Per-topic message-count comparison between rosbag run-dirs, optionally extended with
KITTI-side conversion stats.
"""
import argparse
import csv
import json
import os

from tabulate import tabulate

from headless_bag_reannotation import bag_topics

HEADERS = ["Topic", "Original Count", "Reannotated Count", "Delta",
           "KITTI Frames", "Coverage", "Status"]


def _load_conversion_stats(kitti_dir):
    stats_path = os.path.join(kitti_dir, "conversion_stats.json")
    with open(stats_path) as f:
        return json.load(f)


def build_report(original_dir, new_dir=None, kitti_dir=None):
    """Return rows [[topic, orig_count, new_count, delta, kitti_frames, coverage,
    status], ...] sorted by topic.

    delta = new_count - original_count. A negative delta on a topic that isn't newly
    added means the new bag lost messages relative to the original.
    """
    orig_topics = bag_topics(original_dir)
    new_topics = bag_topics(new_dir) if new_dir else {}
    all_topics = set(orig_topics) | set(new_topics)

    kitti_topics = {}
    if kitti_dir:
        stats = _load_conversion_stats(kitti_dir)
        kitti_topics = stats.get("topics", {})
        all_topics |= set(kitti_topics)

    rows = []
    for topic in sorted(all_topics):
        orig_count = orig_topics.get(topic, 0)
        if new_dir:
            new_count = new_topics.get(topic, 0)
            delta = new_count - orig_count
        else:
            new_count, delta = "-", "-"

        kinfo = kitti_topics.get(topic)
        if kinfo:
            frames_written = kinfo.get("frames_written", "-")
            coverage = kinfo.get("coverage")
            coverage_str = f"{coverage * 100:.1f}%" if coverage is not None else "-"
            status = kinfo.get("status", "-")
        else:
            frames_written, coverage_str, status = "-", "-", "-"

        rows.append([topic, orig_count, new_count, delta, frames_written, coverage_str, status])
    return rows


def write_report(rows, out_prefix):
    """Write `<out_prefix>.md` (tabulate) and `<out_prefix>.csv`."""
    with open(out_prefix + ".md", "w") as f:
        f.write(tabulate(rows, headers=HEADERS, tablefmt="pipe"))
        f.write("\n")
    with open(out_prefix + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", required=True, help="path to the original bag run-dir")
    parser.add_argument("--new", default=None,
                        help="path to the reannotated/merged bag run-dir, if any")
    parser.add_argument("--kitti", default=None,
                        help="path to a converted KITTI dataset dir (reads conversion_stats.json)")
    parser.add_argument("--out", default=None,
                        help="output path prefix; writes <out>.md and <out>.csv (default: print only)")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_report(args.original, args.new, args.kitti)
    print(tabulate(rows, headers=HEADERS, tablefmt="pipe"))

    if args.new:
        dropped = [r for r in rows if isinstance(r[3], int) and r[3] < 0]
        if dropped:
            print("\nWARNING: topics with fewer messages in the new bag (possible data loss):")
            for topic, orig, new, delta, *_ in dropped:
                print(f"  {topic}: {orig} -> {new} ({delta})")

    if args.out:
        write_report(rows, args.out)
        print(f"\nWrote {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
