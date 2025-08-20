import os
import shutil
import pytest

import copy
import torch
torch.manual_seed(37)
import numpy as np

from ros_torch_converter.converter import str_to_cvt_class

"""
Basically, just check that:
    1. data = from_kitti(to_kitti(data))

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