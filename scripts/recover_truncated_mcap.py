"""Detect and repair truncated MCAP files (recorder SIGKILLed mid-write).

A well-formed MCAP ends with the 8-byte magic ``b"\\x89MCAP0\\r\\n"``. When a
recorder is killed hard, the final ``.mcap`` loses the whole trailing section it
writes on clean shutdown -- ``DataEnd`` / ``ChunkIndex`` / ``Statistics`` /
``Summary`` / ``Footer`` + magic -- and usually a partial final ``Chunk`` record
as well. ``ros2 bag reindex`` cannot read a footer-less mcap, so the bag fails to
convert even though ~all of its chunks are intact.

``recover_mcap()`` walks the record framing (seek-only, cheap even on 11 GB),
finds the last complete record, and rebuilds a valid file from that prefix --
**in place, on the copy it is handed, never the source bag** -- matching the
``repair_*`` / ``reindex_bag`` policy in ``dataset_pipeline.py``. Preference
order: the Foxglove ``mcap`` CLI (``mcap recover``) if on PATH, else a
pure-Python rebuild via the ``mcap`` package.

Also usable standalone::

    python recover_truncated_mcap.py <bag_dir> [--inspect] [--out <dir>]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

MCAP_MAGIC = b"\x89MCAP0\r\n"

_OPCODE_NAMES = {
    0x01: "Header", 0x02: "Footer", 0x03: "Schema", 0x04: "Channel",
    0x05: "Message", 0x06: "Chunk", 0x07: "MessageIndex", 0x08: "ChunkIndex",
    0x09: "Attachment", 0x0A: "AttachmentIndex", 0x0B: "Statistics",
    0x0C: "Metadata", 0x0D: "MetadataIndex", 0x0E: "SummaryOffset",
    0x0F: "DataEnd",
}


def mcap_has_trailing_magic(path):
    """Cheap well-formedness gate: True iff the last 8 bytes are the MCAP magic."""
    size = os.path.getsize(path)
    if size < len(MCAP_MAGIC):
        return False
    with open(path, "rb") as f:
        f.seek(-len(MCAP_MAGIC), os.SEEK_END)
        return f.read(len(MCAP_MAGIC)) == MCAP_MAGIC


def walk_mcap_records(path):
    """Seek-only walk of the MCAP record framing from offset 8.

    Returns ``(prefix_len, census, tail_bytes)``: ``prefix_len`` is the byte
    offset just past the last *complete* record, ``census`` is
    ``{record_name: count}`` over those complete records, and ``tail_bytes`` is
    the count of leftover bytes (a partial final record). Raises ``ValueError``
    if the file has no leading MCAP magic.
    """
    size = os.path.getsize(path)
    census = {}
    with open(path, "rb") as f:
        if f.read(len(MCAP_MAGIC)) != MCAP_MAGIC:
            raise ValueError("no leading MCAP magic; not an MCAP file")
        off = len(MCAP_MAGIC)
        while off + 9 <= size:
            f.seek(off)
            opcode = f.read(1)[0]
            rec_len = int.from_bytes(f.read(8), "little")
            rec_end = off + 9 + rec_len
            if rec_end > size:
                break  # partial final record
            name = _OPCODE_NAMES.get(opcode, "op0x%02x" % opcode)
            census[name] = census.get(name, 0) + 1
            off = rec_end
    return off, census, size - off


def _recover_with_cli(src, dst):
    """Rebuild via the Foxglove `mcap` CLI. Returns True on a valid output."""
    if shutil.which("mcap") is None:
        return False
    try:
        subprocess.run(["mcap", "recover", src, "-o", dst],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, check=True)
    except (subprocess.CalledProcessError, OSError):
        return False
    return os.path.exists(dst) and mcap_has_trailing_magic(dst)


def _recover_with_python(src, dst):
    """Rebuild by streaming the source's records through ``mcap.writer``.

    Reads the file sequentially (bounded memory -- no 11 GB slurp) and stops at
    the first record it cannot fully decode, which is the partial tail. Slower
    than the CLI (it re-encodes) but needs no extra binary. Returns True on a
    valid output holding at least one message.
    """
    try:
        from mcap.records import Channel, Header, Message, Schema
        from mcap.stream_reader import StreamReader
        from mcap.writer import Writer
    except ImportError:
        return False

    schema_map, channel_map = {}, {}
    n_msgs = 0
    with open(src, "rb") as in_f, open(dst, "wb") as out_f:
        reader = StreamReader(in_f, skip_magic=False)
        writer = Writer(out_f)
        started = False
        try:
            for rec in reader.records:
                if isinstance(rec, Header):
                    writer.start(profile=rec.profile,
                                 library="recover_truncated_mcap")
                    started = True
                elif isinstance(rec, Schema):
                    if not started:
                        writer.start(library="recover_truncated_mcap")
                        started = True
                    schema_map[rec.id] = writer.register_schema(
                        rec.name, rec.encoding, rec.data)
                elif isinstance(rec, Channel):
                    if not started:
                        writer.start(library="recover_truncated_mcap")
                        started = True
                    channel_map[rec.id] = writer.register_channel(
                        rec.topic, rec.message_encoding,
                        schema_map.get(rec.schema_id, 0), dict(rec.metadata))
                elif isinstance(rec, Message):
                    cid = channel_map.get(rec.channel_id)
                    if cid is None:
                        continue
                    writer.add_message(cid, rec.log_time, rec.data,
                                       rec.publish_time, rec.sequence)
                    n_msgs += 1
        except Exception:
            pass  # truncated tail -- keep every record read so far
        if not started:
            writer.start(library="recover_truncated_mcap")
        writer.finish()
    return n_msgs > 0 and mcap_has_trailing_magic(dst)


def recover_mcap(bag_dir, relpath=""):
    """Repair every truncated ``.mcap`` in ``bag_dir`` in place.

    Intact files (trailing magic present) and 0-byte files (``reindex_bag``
    drops those) are left untouched. Logs ``[recover] <relpath>: ...`` the way
    the sibling ``[repair]`` / ``[reindex]`` helpers do. Returns a list of
    ``(filename, "recovered" | "dropped")`` for the caller to record.
    """
    actions = []
    for name in sorted(os.listdir(bag_dir)):
        if os.path.splitext(name)[1] != ".mcap":
            continue
        path = os.path.join(bag_dir, name)
        if os.path.getsize(path) == 0 or mcap_has_trailing_magic(path):
            continue

        try:
            prefix_len, census, tail = walk_mcap_records(path)
        except ValueError as e:
            print(f"[recover] {relpath}: {name}: {e} -- leaving for reindex to "
                  f"reject", flush=True)
            continue

        size = os.path.getsize(path)
        pct = (100.0 * prefix_len / size) if size else 0.0
        census_str = ", ".join(f"{k}={v}" for k, v in sorted(census.items()))
        print(f"[recover] {relpath}: {name} missing trailing magic -- "
              f"{prefix_len}/{size} bytes ({pct:.2f}%) record-aligned "
              f"[{census_str}]; {tail}-byte partial tail will be dropped",
              flush=True)

        fd, tmp = tempfile.mkstemp(prefix=name + ".recover_", dir=bag_dir)
        os.close(fd)
        how = "mcap recover CLI"
        ok = _recover_with_cli(path, tmp)
        if not ok:
            how = "pure-python rebuild"
            ok = _recover_with_python(path, tmp)

        if ok:
            os.replace(tmp, path)
            print(f"[recover] {relpath}: {name} rebuilt via {how} -> "
                  f"{os.path.getsize(path)} bytes with a valid footer + magic",
                  flush=True)
            actions.append((name, "recovered"))
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
            os.remove(path)
            print(f"[recover] {relpath}: {name} UNRECOVERABLE -- DROPPED. "
                  f"~{prefix_len} bytes of intact chunks lost; bag continues "
                  f"with its remaining storage files.", flush=True)
            actions.append((name, "dropped"))
    return actions


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_dir", help="rosbag2 run-dir holding the .mcap files")
    ap.add_argument("--inspect", action="store_true",
                    help="only report the record walk; do not modify anything")
    ap.add_argument("--out", default=None,
                    help="recover into a copy of bag_dir at this path "
                         "(default: modify bag_dir in place)")
    args = ap.parse_args(argv)

    if args.inspect:
        for name in sorted(os.listdir(args.bag_dir)):
            if not name.endswith(".mcap"):
                continue
            path = os.path.join(args.bag_dir, name)
            size = os.path.getsize(path)
            if size == 0:
                print(f"{name}: 0 bytes (empty)")
                continue
            magic = mcap_has_trailing_magic(path)
            try:
                prefix_len, census, tail = walk_mcap_records(path)
            except ValueError as e:
                print(f"{name}: {size} bytes -- {e}")
                continue
            status = "OK" if magic else "TRUNCATED"
            census_str = ", ".join(f"{k}={v}" for k, v in sorted(census.items()))
            print(f"{name}: {size} bytes, trailing_magic={magic} [{status}]")
            print(f"    record-aligned prefix {prefix_len} ({100.0*prefix_len/size:.2f}%), "
                  f"partial tail {tail} bytes")
            print(f"    records: {census_str}")
        return 0

    target = args.bag_dir
    if args.out:
        if os.path.exists(args.out):
            sys.exit(f"--out {args.out} already exists; refusing to overwrite")
        shutil.copytree(args.bag_dir, args.out)
        target = args.out
        print(f"[recover] copied {args.bag_dir} -> {target}")

    actions = recover_mcap(target, os.path.basename(os.path.normpath(target)))
    if not actions:
        print("[recover] nothing to do -- all .mcap files already well-formed")
    else:
        for name, what in actions:
            print(f"[recover] {name}: {what}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
