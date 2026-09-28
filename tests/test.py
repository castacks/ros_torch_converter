import os
import shutil
import pytest

import copy
import torch
torch.manual_seed(37)
import numpy as np

from ros_torch_converter.converter import str_to_cvt_class
from ros_torch_converter.datatypes.transform import TransformTorch
from ros_torch_converter.tf_manager import TfManager, TfGraph
from tartandriver_utils.geometry_utils import htm_to_pose

"""
Basically, just check that:
    1. data = from_kitti(to_kitti(data))
    2. transform lookups are invariant to tree root 
        (i.e. the root of the transform tree doesn't matter)

(rosmsg cvt is allowed to be lossy because of stricter dtype/interface rules)
"""
@pytest.fixture(scope="session", autouse=True)
def setup():
    os.makedirs('test_data')
    for k in str_to_cvt_class.keys():
        os.makedirs(os.path.join('test_data', k))

    yield

    print('cleaning up test data...')
    shutil.rmtree('test_data')

def test_dtypes():
    for k, dtype_cls in str_to_cvt_class.items():
        print(f'testing {k}...')

        ddir = os.path.join('test_data', k)

        print(dtype_cls)

        for i in range(100):
            data = dtype_cls.rand_init()

            data.to_kitti(ddir, i)
            data2 = dtype_cls.from_kitti(ddir, i)

            assert data == data2

def random_transform(device='cpu'):
    transform = TransformTorch.rand_init(device='cpu').transform.numpy()
    return htm_to_pose(transform)

def build_random_graph(seed, num_frames=10):
    rng = np.random.default_rng(seed)

    graph = TfGraph({})
    frames = [f"frame_{i}" for i in range(num_frames)]

    # Build a random (acyclic) graph of transforms
    for i in range(1, num_frames):
        parent = frames[rng.integers(0, i)]

        if rng.random() < 0.5:
            # Dynamic edge
            times = np.arange(5, dtype=float)
            transforms = np.array([random_transform() for _ in times])

            graph.add_tf(
                src_frame=frames[i],
                dst_frame=parent,
                transforms=transforms,
                times=times,
            )
        else:
            # Static edge
            graph.add_static_tf(
                src_frame=frames[i],
                dst_frame=parent,
                transform=random_transform(),
            )
    has_cycle = graph.validate_graph()
    assert not has_cycle, "Generated graph has cycles!"
    return graph, frames

def test_tree_root_invariance():
    ref_tf_manager = TfManager(device='cpu')
    test_tf_manager = TfManager(device='cpu')

    for seed in range(10):
        graph, frames = build_random_graph(seed)
        ref_tf_manager.tf_graph = graph
        test_tf_manager.tf_graph = graph

        reference_tree = graph.create_tree_with_root(frames[0])
        ref_tf_manager.tf_tree = reference_tree

        print(f"Reference tree with root {frames[0]}: \n {reference_tree}")
        for root in frames[1:]:
            tree = graph.create_tree_with_root(root)
            test_tf_manager.tf_tree = tree
            print(f"Comparing reference tree with root {frames[0]} to tree with root {root}: \n {tree}")
            for frame1 in frames:
                for frame2 in frames:
                    ref_valid_times = ref_tf_manager.get_valid_times(frame1, frame2)
                    test_valid_times = test_tf_manager.get_valid_times(frame1, frame2)

                    np.testing.assert_array_equal(
                        ref_valid_times,
                        test_valid_times,
                        err_msg=(
                            f"Valid times differ "
                            f"(seed={seed}, root={root}, "
                            f"{frame1}->{frame2})"
                        ),
                    )

                    if (
                        np.isneginf(ref_valid_times[0])
                        and np.isposinf(ref_valid_times[1])
                    ):
                        # Static path: can evaluate at any finite timestamp
                        query_times = [0.0]
                    else:
                        query_times = ref_valid_times

                    for t in query_times:
                        ref_tf = ref_tf_manager.get_transform(frame1, frame2, t).transform.numpy()
                        test_tf = test_tf_manager.get_transform(frame1, frame2, t).transform.numpy()

                        # Different tree roots can result in different transform
                        # composition orders, leading to small float32 numerical
                        # differences, hence we use assert_allclose with a small 
                        # tolerance instead of assert_array_equal.
                        np.testing.assert_allclose(
                            ref_tf,
                            test_tf,
                            rtol=1e-6,
                            atol=1e-6,
                            err_msg=(
                                f"Transform mismatch "
                                f"(seed={seed}, root={root}, "
                                f"{frame1}->{frame2}, t={t})"
                            ),
                        )