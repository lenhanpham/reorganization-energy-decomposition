import numpy as np
from enum import Enum
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class SymmetryOperation:
    """Represents a symmetry operation matrix and its properties"""
    matrix: np.ndarray
    name: str
    order: int  # Order of operation (how many times to apply to get identity)

class PointGroup(Enum):
    """Point groups matching Fortran data nmptg"""
    C1 = 1   # No symmetry
    CS = 2   # Mirror plane
    C2 = 3   # 2-fold rotation
    C2V = 4  # 2-fold rotation + mirror plane
    D2H = 5  # 3 2-fold rotations + inversion
    CI = 6   # Inversion center
    C2H = 7  # 2-fold rotation + inversion
    D2 = 8   # 3 2-fold rotations

class SymmetryOperator:
    """Symmetry operations implementation based on Fortran code"""
    
    # Tolerance for geometric comparisons
    TOL = 1.0e-6
    
    # Basic symmetry operations (matching Fortran implementation)
    OPERATIONS = {
        1: SymmetryOperation(  # C2z
            matrix=np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            name="C2z",
            order=2
        ),
        2: SymmetryOperation(  # C2y
            matrix=np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            name="C2y",
            order=2
        ),
        3: SymmetryOperation(  # C2x
            matrix=np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            name="C2x",
            order=2
        ),
        4: SymmetryOperation(  # inversion
            matrix=np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            name="i",
            order=2
        ),
        5: SymmetryOperation(  # σxy
            matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            name="σxy",
            order=2
        ),
        6: SymmetryOperation(  # σxz
            matrix=np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            name="σxz",
            order=2
        ),
        7: SymmetryOperation(  # σyz
            matrix=np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            name="σyz",
            order=2
        )
    }

    def _check_operator(self, coordinates: np.ndarray, 
                       atomic_numbers: np.ndarray, op_index: int) -> bool:
        """
        Check if a symmetry operation is present
        Based on Fortran search routine
        """
        operation = self.OPERATIONS[op_index]
        n_atoms = len(atomic_numbers)
        
        # Apply symmetry operation to all coordinates
        transformed_coords = np.dot(coordinates, operation.matrix.T)
        
        # For each atom in original structure
        for i in range(n_atoms):
            atom_i = atomic_numbers[i]
            coord_i = transformed_coords[i]
            
            # Find matching atom in transformed structure
            found_match = False
            for j in range(n_atoms):
                if atomic_numbers[j] != atom_i:
                    continue
                    
                # Check if coordinates match within tolerance
                diff = np.linalg.norm(coord_i - coordinates[j])
                if diff < self.TOL:
                    found_match = True
                    break
                    
            if not found_match:
                return False
                
        return True

    def _determine_point_group(self, operators: List[bool], n_operators: int) -> PointGroup:
        """Complete point group determination logic"""
        if n_operators == 0:
            return PointGroup.C1
        elif n_operators == 7:
            return PointGroup.D2H
            
        # Check for D2
        if operators[0] and operators[1] and operators[2]:
            return PointGroup.D2
            
        # Check for C2H
        if operators[0] and operators[3]:  # C2z and inversion
            return PointGroup.C2H
            
        # Check for C2V
        if ((operators[0] and operators[5] and operators[6]) or  # C2z and mirrors
            (operators[1] and operators[4] and operators[6]) or  # C2y and mirrors
            (operators[2] and operators[4] and operators[5])):   # C2x and mirrors
            return PointGroup.C2V
            
        # Single operator cases
        if n_operators == 1:
            if any(operators[4:7]):  # Any mirror plane
                return PointGroup.CS
            elif operators[3]:  # Inversion
                return PointGroup.CI
            elif any(operators[0:3]):  # Any C2
                return PointGroup.C2
                
        return PointGroup.C1

    def _adjust_coordinates(self, coordinates: np.ndarray, 
                          operators: List[bool], 
                          point_group: PointGroup) -> np.ndarray:
        """
        Adjust coordinates to standard orientation
        Based on aswap Fortran subroutine
        """
        transform = np.eye(3)
        
        if point_group == PointGroup.C1:
            return transform
            
        # Standard orientations for different point groups
        if point_group == PointGroup.CS:
            # Put mirror plane in xy
            if operators[6]:  # σyz -> xy
                transform = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            elif operators[5]:  # σxz -> xy
                transform = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                
        elif point_group == PointGroup.C2:
            # Put C2 axis along z
            if operators[1]:  # C2y -> z
                transform = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            elif operators[2]:  # C2x -> z
                transform = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                
        elif point_group == PointGroup.C2V:
            # Standard orientation: C2 along z, σv in xz and yz
            if operators[1]:  # C2y is present
                transform = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            elif operators[2]:  # C2x is present
                transform = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                
        return transform

    def _calculate_principal_axes(self, coordinates: np.ndarray, 
                                masses: np.ndarray) -> np.ndarray:
        """
        Calculate principal axes of inertia
        Based on inertm Fortran subroutine
        """
        # Calculate center of mass
        total_mass = np.sum(masses)
        com = np.sum(coordinates * masses[:, np.newaxis], axis=0) / total_mass
        
        # Translate to center of mass
        coords_com = coordinates - com
        
        # Calculate inertia tensor
        inertia = np.zeros((3, 3))
        for i in range(len(masses)):
            x, y, z = coords_com[i]
            inertia[0, 0] += masses[i] * (y*y + z*z)
            inertia[1, 1] += masses[i] * (x*x + z*z)
            inertia[2, 2] += masses[i] * (x*x + y*y)
            inertia[0, 1] -= masses[i] * x * y
            inertia[0, 2] -= masses[i] * x * z
            inertia[1, 2] -= masses[i] * y * z
            
        inertia[1, 0] = inertia[0, 1]
        inertia[2, 0] = inertia[0, 2]
        inertia[2, 1] = inertia[1, 2]
        
        # Get eigenvectors (principal axes)
        eigenvals, eigenvects = np.linalg.eigh(inertia)
        
        return eigenvects

    def standardize_orientation(self, coordinates: np.ndarray, 
                              atomic_numbers: np.ndarray,
                              masses: np.ndarray) -> Tuple[np.ndarray, PointGroup]:
        """
        Main method to detect symmetry and standardize orientation
        
        Args:
            coordinates: (n_atoms, 3) array of atomic coordinates
            atomic_numbers: (n_atoms,) array of atomic numbers
            masses: (n_atoms,) array of atomic masses
            
        Returns:
            Tuple of (transformed_coordinates, point_group)
        """
        # First align to principal axes
        principal_axes = self._calculate_principal_axes(coordinates, masses)
        coords_aligned = np.dot(coordinates, principal_axes)
        
        # Find symmetry elements
        operators = self._find_symmetry_operators(coords_aligned, atomic_numbers)
        n_operators = sum(1 for op in operators if op)
        
        # Determine point group
        point_group = self._determine_point_group(operators, n_operators)
        
        # Get final transformation
        transform = self._adjust_coordinates(coords_aligned, operators, point_group)
        
        # Apply final transformation
        final_coords = np.dot(coords_aligned, transform)
        
        return final_coords, point_group
