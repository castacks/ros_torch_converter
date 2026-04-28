import os
import torch
import numpy as np

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType, TimeSpec
from ros_torch_converter.utils import update_info_file, update_timestamp_file, read_info_file, read_timestamp_file

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from control_interfaces.msg import MPPISolution

from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp

class MPPISolutionTorch(TorchCoordinatorDataType):
    """
    Coordinator type for MPC Solutions.
    """
    to_rosmsg_type = MPPISolution
    from_rosmsg_type = MPPISolution
    time_spec = TimeSpec.SYNC

    def __init__(self, device='cpu'):
        super().__init__()
        self.stamp = None
        self.frame_id = None
        self.state_dim = None
        self.state_keys = None
        self.control_dim = None
        self.control_keys = None
        self.h = None
        self.dt = None
        self.k = None
        self.solution_cost = None
        self.solution_feasible = None
        self.cost_terms = None
        self.solution_term_cost = None
        self.solution_term_feasible = None
        self.solution_controls = None
        self.solution_states = None
        self.random_controls = None
        self.random_states = None
        self.device = device

    def from_torch(
            state_dim,
            state_keys,
            control_dim,
            control_keys,
            h,
            dt,
            solution_cost,
            solution_feasible,
            cost_terms,
            solution_term_cost,
            solution_term_feasible,
            solution_states,
            solution_controls,
            k=0,
            random_states=None,
            random_controls=None,
        ):
        """
        Args:
            state_dim: int of state dimension
            state_keys: list[str] of state variables (len=state_dim)
            control_dim: int of control dimension
            control_keys: list[str] of control variables (len=control_dim)
            h: int of MPPI horizon
            dt: float of MPPI tiem discretization
            solution_cost: float of optimal solution cost
            solution_feasible: bool of solution feasibility
            cost_terms: list[str] of cost terms
            solution_term_cost: FloatTensor of solution cost per term
            solution_term_feasible: BoolTensor of solution feasibility per term
            solution_states: HxN FloatTensor of solution states
            solution_controls: HxM FloatTensor of solution controls
            k: (optional) int of random trajs for viz
            random_states: (optional) KxHxM FloatTensor of random states
            random_controls: (optional) KxHxN FloatTensor of random controls
        """
        ## validate shapes
        assert state_dim == len(state_keys) == solution_states.shape[-1]
        assert control_dim == len(control_keys) ==  solution_controls.shape[-1]
        assert len(cost_terms) == solution_term_cost.shape[0] == solution_term_feasible.shape[0]
        assert h == solution_states.shape[0] == solution_controls.shape[0]

        if k > 0:
            assert k == random_states.shape[0] == random_controls.shape[0]
            assert h == random_states.shape[1] == random_controls.shape[1]
            assert state_dim == random_states.shape[-1]
            assert control_dim == random_controls.shape[-1]

        device = solution_controls.device

        soln = MPPISolutionTorch(device=device)

        soln.state_dim = state_dim
        soln.state_keys = state_keys
        soln.control_dim = control_dim
        soln.control_keys = control_keys
        soln.h = h
        soln.dt = dt
        soln.k = k
        soln.solution_cost = solution_cost.item() if isinstance(solution_cost, torch.Tensor) else solution_cost
        soln.solution_feasible = solution_feasible.item() if isinstance(solution_feasible, torch.Tensor) else solution_feasible
        soln.cost_terms = cost_terms
        soln.solution_term_cost = solution_term_cost.float().to(device)
        soln.solution_term_feasible = solution_term_feasible.float().to(device)
        soln.solution_states = solution_states.float().to(device)
        soln.solution_controls = solution_controls.float().to(device)

        if k > 0:
            soln.random_states = random_states.float().to(device)
            soln.random_controls = random_controls.float().to(device)
        else:
            soln.random_states = torch.zeros(0, h, state_dim, dtype=torch.float32, device=device)
            soln.random_controls = torch.zeros(0, h, control_dim, dtype=torch.float32, device=device)

        return soln

    def from_rosmsg(msg, device='cpu'):
        soln = MPPISolutionTorch(device=device)

        soln.stamp = stamp_to_time(msg.header.stamp)
        soln.frame_id = msg.header.frame_id

        # as is
        soln.state_dim = msg.state_dim
        soln.state_keys = msg.state_keys
        soln.control_dim = msg.control_dim
        soln.control_keys = msg.control_keys
        soln.h = msg.h
        soln.dt = msg.dt
        soln.k = msg.k
        soln.solution_cost = msg.solution_cost
        soln.solution_feasible = msg.solution_feasible
        soln.cost_terms = msg.cost_terms
        soln.solution_term_cost = torch.tensor(msg.solution_term_cost, dtype=torch.float32, device=device)
        soln.solution_term_feasible = msg.solution_term_feasible

        # some reshaping
        soln.solution_states = torch.tensor(msg.solution_states.data.reshape(soln.h, soln.state_dim), dtype=torch.float32, device=device)
        soln.solution_controls = torch.tensor(msg.solution_controls.data.reshape(soln.h, soln.control_dim), dtype=torch.float32, device=device)
        soln.random_states = torch.tensor(msg.random_states.data.reshape(soln.k, soln.h, soln.state_dim), dtype=torch.float32, device=device)
        soln.random_controls = torch.tensor(msg.random_controls.data.reshape(soln.k, soln.h, soln.control_dim), dtype=torch.float32, device=device)
        return soln

    def to_rosmsg(self):
        return None

    def to_kitti(self, base_dir, idx):
        update_timestamp_file(base_dir, idx, self.stamp)
        update_info_file(base_dir, 'frame_id', self.frame_id)

        data = {
            'state_dim': self.state_dim,
            'state_keys': self.state_keys,
            'control_dim': self.control_dim,
            'control_keys': self.control_keys,
            'h': self.h,
            'dt': self.dt,
            'k': self.k,
            'solution_cost': self.solution_cost,
            'solution_feasible': self.solution_feasible,
            'cost_terms': self.cost_terms,
            'solution_term_cost': self.solution_term_cost.cpu().numpy(),
            'solution_term_feasible': self.solution_term_feasible.cpu().numpy(),
            'solution_controls': self.solution_controls.cpu().numpy(),
            'solution_states': self.solution_states.cpu().numpy(),
            'random_controls': self.random_controls.cpu().numpy(),
            'random_states': self.random_states.cpu().numpy(),
        }

        save_fp = os.path.join(base_dir, "{:08d}_data.npz".format(idx))
        np.savez(save_fp, **data)

    def from_kitti(base_dir, idx, device='cpu'):
        fp = os.path.join(base_dir, "{:08d}_data.npz".format(idx))
        data = dict(np.load(fp))

        ## move tensor data to torch
        data['solution_term_cost'] = torch.tensor(data['solution_term_cost'], dtype=torch.float32, device=device)
        data['solution_term_feasible'] = torch.tensor(data['solution_term_feasible'], dtype=torch.bool, device=device)
        data['solution_states'] = torch.tensor(data['solution_states'], dtype=torch.float32, device=device)
        data['solution_controls'] = torch.tensor(data['solution_controls'], dtype=torch.float32, device=device)
        data['random_states'] = torch.tensor(data['random_states'], dtype=torch.float32, device=device)
        data['random_controls'] = torch.tensor(data['random_controls'], dtype=torch.float32, device=device)

        soln = MPPISolutionTorch.from_torch(**data)

        soln.stamp = read_timestamp_file(base_dir, idx)
        soln.frame_id = read_info_file(base_dir,  'frame_id')

        return soln

    def rand_init(device='cpu'):
        import random, string
        k = np.random.randint(11)
        h = np.random.randint(101)
        dt = np.random.randint(1,11)/10
        n = np.random.randint(1,10)
        m = np.random.randint(1,4)
        nterms = np.random.randint(1,8)
        term_cost = torch.rand(nterms, device=device)*10
        term_feas = torch.randint(0,2,size=(nterms,), device=device).bool()

        data = {
            'state_dim': n,
            'state_keys': ["".join(random.sample(string.ascii_letters, 5)) for _ in range(n)],
            'control_dim': m,
            'control_keys': ["".join(random.sample(string.ascii_letters, 5)) for _ in range(m)],
            'h': h,
            'dt': dt,
            'k': k,
            'solution_cost': term_cost.sum(),
            'solution_feasible': term_feas.all(),
            'cost_terms': ["".join(random.sample(string.ascii_letters, 5)) for _ in range(nterms)],
            'solution_term_cost': term_cost,
            'solution_term_feasible': term_feas,
            'solution_controls': torch.rand(h,m, device=device),
            'solution_states': torch.rand(h,n, device=device),
            'random_controls': torch.rand(k,h,m, device=device),
            'random_states': torch.rand(k,h,n, device=device),
        }
        soln = MPPISolutionTorch.from_torch(**data)

        soln.frame_id = 'random'
        soln.stamp = np.random.rand()

        return soln

    def __eq__(self, other):
        if self.frame_id != other.frame_id:
            return False

        if abs(self.stamp - other.stamp) > 1e-8:
            return False

        if not self.state_dim==other.state_dim:
            return False
        if not all([k1==k2 for k1, k2 in zip(self.state_keys, other.state_keys)]):
            return False
        if not self.control_dim==other.control_dim:
            return False
        if not all([k1==k2 for k1, k2 in zip(self.control_keys, other.control_keys)]):
            return False
        if not self.h==other.h:
            return False
        if not self.dt==other.dt:
            return False
        if not self.k==other.k:
            return False
        if not abs(self.solution_cost-other.solution_cost)<1e-8:
            return False
        if not self.solution_feasible==other.solution_feasible:
            return False
        if not all([k1==k2 for k1, k2 in zip(self.cost_terms, other.cost_terms)]):
            return False
        if not torch.allclose(self.solution_term_cost, other.solution_term_cost):
            return False
        if not (self.solution_term_feasible==other.solution_term_feasible).all():
            return False
        if not torch.allclose(self.solution_controls, other.solution_controls):
            return False
        if not torch.allclose(self.solution_states, other.solution_states):
            return False
        if not torch.allclose(self.random_controls, other.random_controls):
            return False
        if not torch.allclose(self.random_states, other.random_states):
            return False

        return True

    def to(self, device):
        self.device = device
        self.cost_terms = self.cost_terms.to(device)
        self.solution_states = self.solution_states.to(device)
        self.solution_controls = self.solution_controls.to(device)
        self.random_states = self.random_states.to(device)
        self.random_controls = self.random_controls.to(device)
        return self

    def __repr__(self):
        return (
            "MPPISolutionTorch with soln cost={}, soln feas={}, h={}, dt={}, k={}, N={}, M={}, device {}".format(
                self.solution_cost, self.solution_feasible, self.h, self.dt, self.k, self.state_dim, self.control_dim, self.device
            )
        )
