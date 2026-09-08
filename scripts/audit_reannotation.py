"""Audit original-topic retention and thermal input/output header timestamps."""
import argparse
from collections import Counter
import json
from pathlib import Path

import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def topic_counts(directory):
    info = yaml.safe_load((Path(directory) / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    return {row['topic_metadata']['name']: row['message_count']
            for row in info['topics_with_message_count']}


def header_stamps(directory, topics):
    result = {topic: Counter() for topic in topics}
    with AnyReader([Path(directory)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [connection for connection in reader.connections if connection.topic in result]
        for connection, _, data in reader.messages(connections=connections):
            message = reader.deserialize(data, connection.msgtype)
            stamp = message.header.stamp
            result[connection.topic][stamp.sec * 1_000_000_000 + stamp.nanosec] += 1
    return result


def audit(source, merged, minimum_fraction=0.995, remap_prefix='/prereannotation', forced_topics=(), excluded_topics=(), keep_remapped=True):
    original, final = topic_counts(source), topic_counts(merged)
    original_losses = []
    for topic, count in original.items():
        if topic in excluded_topics or (topic in forced_topics and not keep_remapped):
            continue
        destination = remap_prefix.rstrip('/') + topic if topic in forced_topics else topic
        if final.get(destination, 0) < count:
            original_losses.append({'topic': topic, 'destination': destination,
                                    'original': count, 'retained': final.get(destination, 0)})
    pairs = []
    for side in ('left', 'right'):
        raw = f'/thermal_{side}/image'
        if original.get(raw, 0):
            pairs.extend((raw, raw + suffix) for suffix in ('_processed', '_processed/camera_info'))
    source_stamps = header_stamps(source, {raw for raw, _ in pairs})
    output_stamps = header_stamps(merged, {processed for _, processed in pairs})
    thermal = []
    for raw, processed in pairs:
        expected = source_stamps[raw]
        observed = output_stamps[processed]
        matched = sum((expected & observed).values())
        total = sum(expected.values())
        missing = expected - observed
        thermal.append({'input_topic': raw, 'output_topic': processed, 'input_frames': total,
                        'output_frames': sum(observed.values()), 'matched_stamps': matched,
                        'missing_frames': total - matched, 'retention': matched / total if total else 1,
                        'first_missing_stamps_ns': list(sorted(missing))[:20]})
    return {'source': str(source), 'merged': str(merged), 'minimum_thermal_retention': minimum_fraction,
            'original_topic_losses': original_losses, 'excluded_original_topics': sorted(excluded_topics),
            'thermal': thermal,
            'passed': not original_losses and all(row['retention'] >= minimum_fraction for row in thermal)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--merged', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--minimum-fraction', type=float, default=0.995)
    parser.add_argument('--forced-thermal', action='store_true')
    args = parser.parse_args()
    forced = [f'/thermal_{side}/image_processed{suffix}' for side in ('left', 'right')
              for suffix in ('', '/camera_info')] if args.forced_thermal else []
    result = audit(args.source, args.merged, args.minimum_fraction, forced_topics=forced)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
