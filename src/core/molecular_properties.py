import numpy as np
from typing import Tuple, Optional

class MolecularProperties:
    """Handles molecular property calculations"""
    
    def __init__(self):
        self.amu_to_au = 1822.888486  # Mass conversion factor
        
    def calculate_inertia_tensor(self,
                               coords: np.ndarray,
                               masses: np.ndarray,
                               center: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate moment of inertia tensor
        
        Args:
            coords: Cartesian coordinates (N x 3)
            masses: Atomic masses (N)
            center: Whether to center coordinates at COM
            
        Returns:
            Tuple containing:
            - Inertia tensor
            - Centered coordinates (if center=True)
        """
        if center:
            # Calculate center of mass
            com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
            coords = coords - com
            
        # Calculate inertia tensor
        inertia = np.zeros((3, 3))
        for i in range(len(masses)):
            x, y, z = coords[i]
            inertia[0,0] += masses[i] * (y*y + z*z)
            inertia[1,1] += masses[i] * (x*x + z*z)
            inertia[2,2] += masses[i] * (x*x + y*y)
            inertia[0,1] -= masses[i] * x * y
            inertia[0,2] -= masses[i] * x * z
            inertia[1,2] -= masses[i] * y * z
            
        # Symmetrize
        inertia[1,0] = inertia[0,1]
        inertia[2,0] = inertia[0,2]
        inertia[2,1] = inertia[1,2]
        
        return inertia, coords
    
    def diagonalize_inertia(self,
                           coords: np.ndarray,
                           masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate and diagonalize inertia tensor
        
        Args:
            coords: Cartesian coordinates
            masses: Atomic masses
            
        Returns:
            Tuple containing:
            - Principal moments
            - Rotation matrix
            - Transformed coordinates
        """
        # Calculate inertia tensor
        inertia, coords_centered = self.calculate_inertia_tensor(coords, masses)
        
        # Diagonalize
        moments, rot_matrix = np.linalg.eigh(inertia)
        
        # Transform coordinates
        coords_transformed = np.dot(coords_centered, rot_matrix)
        
        return moments, rot_matrix, coords_transformed