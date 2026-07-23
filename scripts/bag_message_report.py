"""Per-topic message-count comparison between two rosbag run-dirs.

Meant to check that reannotation didn't lose any original data: compare a bag before
reannotation against its merged counterpart (see headless_bag_reannotation.py) and flag
any topic whose count dropped.
"""
import argparse
import csv

from tabulate import tabulate

from headless_bag_reannotation import bag_topics

HEADERS = ["Topic", "Original Count", "New Count", "Delta"]


def build_report(original_dir, new_dir):
    """Return rows [[topic, orig_count, new_count, delta], ...] sorted by topic.

    delta = new_count - original_count. A negative delta on a topic that isn't newly
    added means the new bag lost messages relative to the original.
    """
    orig_topics = bag_topics(original_dir)
    new_topics = bag_topics(new_dir)
    all_topics = sorted(set(orig_topics) | set(new_topics))
    return [
        [topic, orig_topics.get(topic, 0), new_topics.get(topic, 0),
         new_topics.get(topic, 0) - orig_topics.get(topic, 0)]
        for topic in all_topics
    ]


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
    parser.add_argument("--new", required=True, help="path to the new/merged bag run-dir")
    parser.add_argument("--out", default=None,
                        help="output path prefix; writes <out>.md and <out>.csv (default: print only)")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_report(args.original, args.new)
    print(tabulate(rows, headers=HEADERS, tablefmt="pipe"))

    dropped = [r for r in rows if r[3] < 0]
    if dropped:
        print("\nWARNING: topics with fewer messages in the new bag (possible data loss):")
        for topic, orig, new, delta in dropped:
            print(f"  {topic}: {orig} -> {new} ({delta})")

    if args.out:
        write_report(rows, args.out)
        print(f"\nWrote {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
