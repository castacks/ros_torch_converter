import torch

from ros_torch_converter.datatypes.base import TorchCoordinatorDataType
from tartandriver_utils.ros_utils import stamp_to_time, time_to_stamp
from geometry_msgs.msg import Pose

# Import the TrackedHumanArray message (Required)
try:
    from human_tracker_msgs.msg import TrackedHumanArray, TrackedHuman
except ImportError as e:
    raise ImportError(
        "human_tracker_msgs messages not found. You must install the human_tracker_msgs package "
        "to use PeopleDetections. Build the human_tracker_msgs package in your workspace."
    ) from e


class PeopleDetectionsTorch(TorchCoordinatorDataType):
    """
    Torch datatype for people detections from the human_tracker package
    
    Stores 3D positions of detected people along with additional tracking information
    like tracking IDs, velocities (if available), confidence scores, etc.
    
    Message type: TrackedHumanArray from human_tracker package
    """
    to_rosmsg_type = TrackedHumanArray
    from_rosmsg_type = TrackedHumanArray
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.positions = torch.zeros(0, 3, device=device)  # [N x 3] (x, y, z)
        self.tracking_ids = torch.zeros(0, dtype=torch.int32, device=device)  # [N]
        self.confidence = torch.zeros(0, device=device)  # [N]
        self.sizes = torch.zeros(0, 3, device=device)  # [N x 3] (width, depth, height)
        self.detection_counts = torch.zeros(0, dtype=torch.int32, device=device)  # [N]
        self.position_errors = torch.zeros(0, 3, device=device)  # [N x 3] position std dev
        self.device = device
    
    @staticmethod
    def from_rosmsg(msg, device='cpu'):
        """
        Convert TrackedHumanArray message to PeopleDetectionsTorch
        
        Args:
            msg: TrackedHumanArray message from human_tracker
            device: torch device ('cpu' or 'cuda')
        
        Returns:
            PeopleDetectionsTorch with all tracking data
        """
        res = PeopleDetectionsTorch(device=device)
        
        # Extract data from TrackedHumanArray
        positions = []
        tracking_ids = []
        confidences = []
        sizes = []
        detection_counts = []
        position_errors = []
        
        for human in msg.humans:
            # Extract position from Pose
            positions.append([
                human.pose.position.x,
                human.pose.position.y,
                human.pose.position.z
            ])
            
            # Extract tracking metadata
            tracking_ids.append(human.tracking_id)
            confidences.append(human.confidence)
            detection_counts.append(human.detection_count)
            
            # Extract size [width, depth, height]
            sizes.append(human.size)
            
            # Extract position uncertainty
            position_errors.append(human.position_error)
        
        if positions:
            res.positions = torch.tensor(positions, dtype=torch.float32, device=device)
            res.tracking_ids = torch.tensor(tracking_ids, dtype=torch.int32, device=device)
            res.confidence = torch.tensor(confidences, dtype=torch.float32, device=device)
            res.sizes = torch.tensor(sizes, dtype=torch.float32, device=device)
            res.detection_counts = torch.tensor(detection_counts, dtype=torch.int32, device=device)
            res.position_errors = torch.tensor(position_errors, dtype=torch.float32, device=device)
        
        res.stamp = stamp_to_time(msg.header.stamp)
        res.frame_id = msg.header.frame_id
        
        return res
    
    @staticmethod
    def from_numpy(positions, device='cpu'):
        """
        Create from numpy array
        Args:
            positions: [N x 3] numpy array of 3D positions
        """
        res = PeopleDetectionsTorch(device=device)
        res.positions = torch.tensor(positions, dtype=torch.float32, device=device)
        return res
    
    @staticmethod
    def from_torch(positions):
        """
        Create from torch tensor
        Args:
            positions: [N x 3] tensor of 3D positions
        """
        res = PeopleDetectionsTorch(device=positions.device)
        res.positions = positions.float()
        return res
    
    def to_rosmsg(self):
        """Convert to TrackedHumanArray message"""
        msg = TrackedHumanArray()
        
        if hasattr(self, 'stamp'):
            msg.header.stamp = time_to_stamp(self.stamp)
        if hasattr(self, 'frame_id'):
            msg.header.frame_id = self.frame_id
        
        positions_np = self.positions.cpu().numpy()
        tracking_ids_np = self.tracking_ids.cpu().numpy()
        confidences_np = self.confidence.cpu().numpy()
        sizes_np = self.sizes.cpu().numpy()
        detection_counts_np = self.detection_counts.cpu().numpy()
        position_errors_np = self.position_errors.cpu().numpy()
        
        for i, pos in enumerate(positions_np):
            human = TrackedHuman()
            human.tracking_id = int(tracking_ids_np[i])
            human.confidence = float(confidences_np[i])
            human.detection_count = int(detection_counts_np[i])
            
            # Position
            human.pose = Pose()
            human.pose.position.x = float(pos[0])
            human.pose.position.y = float(pos[1])
            human.pose.position.z = float(pos[2])
            human.pose.orientation.w = 1.0  # Identity orientation (Default)
            
            # Size and errors
            human.size = sizes_np[i].tolist()
            human.position_error = position_errors_np[i].tolist()
            human.size_error = [0.0, 0.0, 0.0]  # Default
            human.shape_type = 0  # Default (TARGET_SHAPE_CUBE)
            
            msg.humans.append(human)
        
        return msg
    
    def to(self, device):
        """Move data to device"""
        self.device = device
        self.positions = self.positions.to(device)
        self.tracking_ids = self.tracking_ids.to(device)
        self.confidence = self.confidence.to(device)
        self.sizes = self.sizes.to(device)
        self.detection_counts = self.detection_counts.to(device)
        self.position_errors = self.position_errors.to(device)
        return self
    
    def __len__(self):
        """Return number of detections"""
        return self.positions.shape[0]
    
    @staticmethod
    def from_kitti(kitti_dict, device='cpu'):
        """
        Create from KITTI format dictionary (not applicable for people detections)
        
        This method is required by the base class but not used for people detections.
        Returns an empty PeopleDetectionsTorch object.
        """
        return PeopleDetectionsTorch(device=device)
    
    def to_kitti(self):
        """
        Convert to KITTI format (not applicable for people detections)
        
        This method is required by the base class but not used for people detections.
        Returns an empty dictionary.
        """
        return {}
