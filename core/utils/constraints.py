import numpy as np
from typing import Dict, List, Union
from scipy.sparse import csr_matrix
from core.models.radar_network import RadarNetwork


class ConstraintChecker:
    """
    Constraint checker class for validating radar network scheduling constraints:
    1. Each target can only be tracked by one radar at a time (C13).
    2. The number of targets tracked by a radar must not exceed its channel count (C14).
    3. Targets must be within the radar's coverage range (C15).
    4. Assignment matrix entries must be binary (C16).
    """

    @staticmethod
    def check_single_assignment(assignment: csr_matrix) -> bool:
        """
        C13 Constraint: Ensure each target is tracked by at most one radar at any given time.
        """
        return np.all(assignment.sum(axis=1).A1 <= 1)

    @staticmethod
    def check_radar_channels(assignment: csr_matrix, radar_network: RadarNetwork) -> bool:
        """
        C14 Constraint: Ensure the number of targets tracked by each radar does not exceed its available channels.
        """
        allocated_targets = assignment.sum(axis=0).A1
        radar_ids = list(radar_network.radars.keys())

        for i, radar_id in enumerate(radar_ids):
            if i < len(allocated_targets):
                radar = radar_network.radars[radar_id]
                if allocated_targets[i] > radar.num_channels:
                    return False
        return True

    @staticmethod
    def check_radar_coverage(assignment: csr_matrix, radar_network: RadarNetwork, target_positions: List[np.ndarray]) -> bool:
        """
        C15 Constraint: Ensure each assigned target is within the radar's coverage range.
        """
        rows, cols = assignment.nonzero()
        for target_idx, radar_idx in zip(rows, cols):
            if target_idx >= len(target_positions):
                return False
            radar = radar_network.radars.get(radar_idx)
            if radar is None or not radar.is_target_in_range(target_positions[target_idx]):
                return False
        return True

    @staticmethod
    def check_binary_variables(assignment: csr_matrix) -> bool:
        """
        C16 Constraint: Ensure that assignment matrix entries are binary (0 or 1).
        """
        return np.all(np.isin(assignment.data, [0, 1])) if assignment.count_nonzero() > 0 else True

    @staticmethod
    def verify_all_constraints(assignment: csr_matrix, radar_network: RadarNetwork, target_positions: List[np.ndarray]) -> Dict[str, Union[bool, List[int]]]:
        """
        Run all constraint checks and return the results.
        Returns a dictionary with:
        - "C13": True/False
        - "C14": List of overloaded radar IDs
        - "C15": List of target IDs outside radar coverage
        - "C16": True/False
        """
        results = {
            "C13": ConstraintChecker.check_single_assignment(assignment),
            "C14": [],
            "C15": [],
            "C16": ConstraintChecker.check_binary_variables(assignment)
        }

        # Check for overloaded radars (C14)
        allocated_targets = assignment.sum(axis=0).A1
        radar_ids = list(radar_network.radars.keys())
        results["C14"] = [radar_id for i, radar_id in enumerate(radar_ids) if i < len(allocated_targets) and allocated_targets[i] > radar_network.radars[radar_id].num_channels]

        # Check for targets outside radar coverage (C15)
        rows, cols = assignment.nonzero()
        results["C15"] = [target_idx for target_idx, radar_idx in zip(rows, cols) if target_idx >= len(target_positions) or not radar_network.radars.get(radar_idx).is_target_in_range(target_positions[target_idx])]

        return results
