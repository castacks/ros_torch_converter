from pathlib import Path

import pytest

from ros_torch_converter.tf_manager import rosbag_paths, rosbag_readers


def test_rosbag_paths_supports_split_ros1_and_ros2(tmp_path):
    first = tmp_path / "run_0.bag"
    second = tmp_path / "run_1.bag"
    second.touch()
    first.touch()
    assert rosbag_paths(tmp_path) == [first, second]

    (tmp_path / "metadata.yaml").touch()
    assert rosbag_paths(tmp_path) == [Path(tmp_path)]


def test_rosbag_paths_rejects_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        rosbag_paths(tmp_path)


def test_rosbag_readers_close_each_split_before_next(monkeypatch, tmp_path):
    paths = [tmp_path / "a.bag", tmp_path / "b.bag"]
    for path in paths:
        path.touch()
    events = []

    class Reader:
        def __init__(self, opened_paths, default_typestore):
            self.path = opened_paths[0]

        def __enter__(self):
            events.append(("open", self.path.name))
            return self

        def __exit__(self, *args):
            events.append(("close", self.path.name))

    monkeypatch.setattr("ros_torch_converter.tf_manager.AnyReader", Reader)
    assert [reader.path for reader in rosbag_readers(tmp_path, object())] == paths
    assert events == [
        ("open", "a.bag"),
        ("close", "a.bag"),
        ("open", "b.bag"),
        ("close", "b.bag"),
    ]
