from types import SimpleNamespace

import pytest

import ros_torch_converter.converter as converter_module
from ros_torch_converter.converter import ROSTorchConverter


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)


class FakeSubscriber:
    def __init__(self, node, msg_type, topic):
        self.node = node
        self.msg_type = msg_type
        self.topic = topic


class FakeSynchronizer:
    def __init__(self, subscribers, queue_size, slop):
        self.subscribers = subscribers
        self.queue_size = queue_size
        self.slop = slop
        self.callback = None

    def registerCallback(self, callback):
        self.callback = callback


def make_converter(monkeypatch):
    topics = [
        {
            "name": "left",
            "group": "mapping",
            "topic": "/thermal_left/image_processed",
        },
        {
            "name": "right",
            "group": "mapping",
            "topic": "/thermal_right/image_processed",
        },
    ]
    logger = FakeLogger()
    converter = SimpleNamespace(
        config={"topics": topics},
        converters={
            "mapping/left": SimpleNamespace(from_rosmsg_type="left_type"),
            "mapping/right": SimpleNamespace(from_rosmsg_type="right_type"),
        },
        subscribers={},
        synchronizers=[],
        synced_topics=set(),
        sync_lock=False,
        data={"mapping/left": None, "mapping/right": None},
        data_times={"mapping/left": -1.0, "mapping/right": -1.0},
        get_logger=lambda: logger,
    )
    converter._resolve_topic_config = lambda topic_ref: (
        ROSTorchConverter._resolve_topic_config(converter, topic_ref)
    )
    converter.handle_synchronized_msgs = lambda msgs, configs: (
        ROSTorchConverter.handle_synchronized_msgs(converter, msgs, configs)
    )

    monkeypatch.setattr(converter_module, "Subscriber", FakeSubscriber)
    monkeypatch.setattr(
        converter_module, "ApproximateTimeSynchronizer", FakeSynchronizer
    )
    monkeypatch.setattr(converter_module, "stamp_to_time", lambda stamp: stamp)
    return converter, logger


@pytest.mark.parametrize(
    "topic_refs",
    [
        ["mapping/left", "mapping/right"],
        ["left", "right"],
    ],
)
def test_synchronized_topics_use_canonical_grouped_keys(monkeypatch, topic_refs):
    converter, logger = make_converter(monkeypatch)

    ROSTorchConverter._setup_synchronized_subscribers(
        converter,
        [{"topics": topic_refs, "queue_size": 2, "slop": 0.02}],
    )

    assert logger.warnings == []
    assert converter.synced_topics == {"mapping/left", "mapping/right"}
    assert set(converter.subscribers) == {"mapping/left", "mapping/right"}
    assert len(converter.synchronizers) == 1

    synchronizer = converter.synchronizers[0]
    assert [subscriber.topic for subscriber in synchronizer.subscribers] == [
        "/thermal_left/image_processed",
        "/thermal_right/image_processed",
    ]

    left = SimpleNamespace(header=SimpleNamespace(stamp=1.0))
    right = SimpleNamespace(header=SimpleNamespace(stamp=2.0))
    synchronizer.callback(left, right)

    assert converter.data == {
        "mapping/left": left,
        "mapping/right": right,
    }
    assert converter.data_times == {
        "mapping/left": 1.0,
        "mapping/right": 2.0,
    }


def test_invalid_sync_group_falls_back_to_ordinary_subscriptions(monkeypatch):
    converter, logger = make_converter(monkeypatch)

    ROSTorchConverter._setup_synchronized_subscribers(
        converter,
        [{"topics": ["mapping/left", "mapping/missing"]}],
    )

    assert converter.synced_topics == set()
    assert converter.subscribers == {}
    assert converter.synchronizers == []
    assert len(logger.warnings) == 2
