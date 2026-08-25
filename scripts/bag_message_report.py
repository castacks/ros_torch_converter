"""Per-topic message-count comparison between rosbag run-dirs, optionally extended with
KITTI-side conversion stats.
"""
import argparse
import csv
import json
import os

from tabulate import tabulate

from headless_bag_reannotation import bag_topics, remapped_topic

HEADERS = ["Topic", "Original Count", "Reannotated Count", "Delta",
           "KITTI Frames", "Coverage", "Status", "Note"]

# Note-column tags for a bag reannotated with `reannotation.force`: the original bag
# already had these topics, so the counts are a fresh run vs the old data, not a merge.
FORCED_TAG = "FORCED REGEN (replaces original)"
PRESERVED_TAG = "pre-reannotation copy of {}"


def _load_conversion_stats(kitti_dir):
    stats_path = os.path.join(kitti_dir, "conversion_stats.json")
    with open(stats_path) as f:
        return json.load(f)


def build_report(original_dir, new_dir=None, kitti_dir=None,
                 forced_topics=(), remap_prefix=""):
    """Return rows [[topic, orig_count, new_count, delta, kitti_frames, coverage,
    status, note], ...] sorted by topic.

    delta = new_count - original_count. A negative delta on a topic that isn't newly
    added means the new bag lost messages relative to the original -- except on a
    forced-regeneration topic, where the two counts come from different sources.

    `forced_topics` (with `remap_prefix`) are the topics the stack regenerates when
    `reannotation.force` is on: those the original bag already had are tagged
    FORCED_TAG, and the copies kept under `remap_prefix` are tagged as such, so the
    table says outright which counts are the fresh run and which are the old data.
    """
    orig_topics = bag_topics(original_dir)
    new_topics = bag_topics(new_dir) if new_dir else {}
    all_topics = set(orig_topics) | set(new_topics)

    notes = {}
    if new_dir:
        for topic in forced_topics:
            if orig_topics.get(topic, 0) > 0:
                notes[topic] = FORCED_TAG
                if remap_prefix:
                    notes[remapped_topic(topic, remap_prefix)] = PRESERVED_TAG.format(topic)

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

        rows.append([topic, orig_count, new_count, delta, frames_written, coverage_str,
                     status, notes.get(topic, "-")])
    return rows


def write_report(rows, out_prefix):
    """Write `<out_prefix>.md` (tabulate) and `<out_prefix>.csv`."""
    forced = [row[0] for row in rows if row[-1] == FORCED_TAG]
    with open(out_prefix + ".md", "w") as f:
        if forced:
            f.write("**Forced reannotation** (`reannotation.force`): {} were regenerated "
                    "by this run -- for those rows the Original Count column is the data "
                    "they replaced, not a baseline to match.\n\n".format(
                        ", ".join("`{}`".format(t) for t in forced)))
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
        # a forced-regen topic's counts are two different runs, so a drop means nothing there
        dropped = [r for r in rows if isinstance(r[3], int) and r[3] < 0 and r[-1] != FORCED_TAG]
        if dropped:
            print("\nWARNING: topics with fewer messages in the new bag (possible data loss):")
            for topic, orig, new, delta, *_ in dropped:
                print(f"  {topic}: {orig} -> {new} ({delta})")

    if args.out:
        write_report(rows, args.out)
        print(f"\nWrote {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
