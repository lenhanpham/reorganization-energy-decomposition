import numpy as np
from typing import List, Tuple, Dict
from enum import Enum
import warnings

class CoordinateType(Enum):
    STRETCH = 1
    BEND = 2
    PROPER_TORSION = 3
    IMPROPER_TORSION = 4

class ScaleFactorCalculator:
    """Handles scale factor calculations for internal coordinates"""
    
    def __init__(self):
        # Standard conversion factors
        self.deg_to_rad = np.pi / 180.0
        self.rad_to_deg = 180.0 / np.pi
        
        # Default scale factors for each coordinate type
        self.default_scales = {
            CoordinateType.STRETCH: 1.0,  # Angstroms
            CoordinateType.BEND: self.deg_to_rad,  # Degrees to radians
            CoordinateType.PROPER_TORSION: self.deg_to_rad,  # Degrees to radians
            CoordinateType.IMPROPER_TORSION: self.deg_to_rad  # Degrees to radians
        }
    
    def calculate_scale_factors(self, 
                              internal_coords: List[Dict],
                              force_constants: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate scale factors for internal coordinates
        
        Args:
            internal_coords: List of internal coordinate definitions
            force_constants: Optional force constants for weighted scaling
            
        Returns:
            Array of scale factors for each coordinate
        """
        n_coords = len(internal_coords)
        scale_factors = np.ones(n_coords)
        
        for i, coord in enumerate(internal_coords):
            # Get basic scale factor based on coordinate type
            basic_scale = self.default_scales[coord['type']]
            
            # Apply force constant weighting if available
            if force_constants is not None:
                fc = force_constants[i]
                if fc > 0:
                    # Weight scale factor by square root of force constant
                    scale_factors[i] = basic_scale * np.sqrt(fc)
                else:
                    scale_factors[i] = basic_scale
            else:
                scale_factors[i] = basic_scale
        
        return scale_factors
    
    def apply_scale_factors(self, 
                          b_matrix: np.ndarray,
                          scale_factors: np.ndarray) -> np.ndarray:
        """
        Apply scale factors to B-matrix
        
        Args:
            b_matrix: Original B-matrix
            scale_factors: Array of scale factors
            
        Returns:
            Scaled B-matrix
        """
        return b_matrix * scale_factors[:, np.newaxis]
    
    def remove_scale_factors(self,
                           b_matrix: np.ndarray,
                           scale_factors: np.ndarray) -> np.ndarray:
        """
        Remove scale factors from B-matrix
        
        Args:
            b_matrix: Scaled B-matrix
            scale_factors: Array of scale factors
            
        Returns:
            Unscaled B-matrix
        """
        return b_matrix / scale_factors[:, np.newaxis]

class InternalCoordinates:
    """Handles internal coordinate transformations and B-matrix operations"""
    
    def __init__(self, coordinates: np.ndarray, atomic_numbers: np.ndarray):
        self.coords = coordinates
        self.atomic_numbers = atomic_numbers
        self.n_atoms = len(atomic_numbers)
        self.scale_calculator = ScaleFactorCalculator()
        
    def calculate_b_matrix_with_scaling(self) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Calculate B-matrix with appropriate scaling factors
        
        Returns:
            Tuple containing:
            - Scaled B-matrix
            - List of internal coordinates
            - Array of scale factors
        """
        # Calculate basic B-matrix and get internal coordinates
        b_matrix, internal_coords = self.calculate_b_matrix()
        
        # Calculate scale factors
        scale_factors = self.scale_calculator.calculate_scale_factors(internal_coords)
        
        # Apply scaling to B-matrix
        b_matrix_scaled = self.scale_calculator.apply_scale_factors(b_matrix, scale_factors)
        
        return b_matrix_scaled, internal_coords, scale_factors
    
    def transform_coordinates_with_scaling(self,
                                        cartesian_disp: np.ndarray,
                                        b_matrix: np.ndarray,
                                        scale_factors: np.ndarray) -> np.ndarray:
        """
        Transform coordinates with proper scaling
        
        Args:
            cartesian_disp: Cartesian displacement vector
            b_matrix: B-matrix (already scaled)
            scale_factors: Scale factors for each coordinate
            
        Returns:
            Properly scaled internal coordinate displacements
        """
        # Transform to internal coordinates
        internal_disp = np.dot(b_matrix, cartesian_disp)
        
        # Convert to appropriate units (e.g., degrees for angles)
        internal_disp_scaled = internal_disp * self.scale_calculator.rad_to_deg
        
        return internal_disp_scaled
    
    def transform_gradients_with_scaling(self,
                                       internal_grad: np.ndarray,
                                       b_matrix: np.ndarray,
                                       scale_factors: np.ndarray) -> np.ndarray:
        """
        Transform gradients with proper scaling
        
        Args:
            internal_grad: Internal coordinate gradients
            b_matrix: B-matrix (already scaled)
            scale_factors: Scale factors for each coordinate
            
        Returns:
            Cartesian gradients
        """
        # Convert units of gradients (e.g., from degrees to radians for angles)
        internal_grad_scaled = internal_grad * self.scale_calculator.deg_to_rad
        
        # Use pseudo-inverse for transformation
        b_inv = np.linalg.pinv(b_matrix)
        return np.dot(b_inv.T, internal_grad_scaled)
    
    def calculate_b_matrix(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Calculate B-matrix (Wilson matrix) for internal coordinates
        Based on bmatred Fortran subroutine
        """
        # First determine connectivity and internal coordinates
        internal_coords = self._determine_internal_coords()
        
        # Initialize B-matrix
        n_internals = len(internal_coords)
        b_matrix = np.zeros((n_internals, 3 * self.n_atoms))
        
        # Fill B-matrix for each internal coordinate
        for i, coord in enumerate(internal_coords):
            coord_type = coord['type']
            atoms = coord['atoms']
            
            if coord_type == CoordinateType.STRETCH:
                self._add_stretch_terms(b_matrix, i, atoms)
            elif coord_type == CoordinateType.BEND:
                self._add_bend_terms(b_matrix, i, atoms)
            elif coord_type == CoordinateType.PROPER_TORSION:
                self._add_torsion_terms(b_matrix, i, atoms)
            elif coord_type == CoordinateType.IMPROPER_TORSION:
                self._add_improper_terms(b_matrix, i, atoms)
        
        return b_matrix, internal_coords
    
    def _determine_internal_coords(self) -> List[Dict]:
        """Determine internal coordinates based on molecular structure"""
        internal_coords = []
        
        # Add bond stretches
        for i in range(self.n_atoms):
            for j in range(i+1, self.n_atoms):
                if self._are_bonded(i, j):
                    internal_coords.append({
                        'type': CoordinateType.STRETCH,
                        'atoms': (i, j)
                    })
        
        # Add angles
        for i in range(self.n_atoms):
            for j in range(self.n_atoms):
                for k in range(j+1, self.n_atoms):
                    if (i != j and i != k and 
                        self._are_bonded(i, j) and self._are_bonded(j, k)):
                        internal_coords.append({
                            'type': CoordinateType.BEND,
                            'atoms': (i, j, k)
                        })
        
        # Add torsions
        for i, j, k, l in self._find_torsions():
            internal_coords.append({
                'type': CoordinateType.PROPER_TORSION,
                'atoms': (i, j, k, l)
            })
        
        return internal_coords
    
    def _are_bonded(self, i: int, j: int) -> bool:
        """Determine if two atoms are bonded based on distance and atomic numbers"""
        dist = np.linalg.norm(self.coords[i] - self.coords[j])
        r1 = self._get_covalent_radius(self.atomic_numbers[i])
        r2 = self._get_covalent_radius(self.atomic_numbers[j])
        return dist < 1.3 * (r1 + r2)  # 1.3 is a typical scaling factor
    
    def _get_covalent_radius(self, atomic_number: int) -> float:
        """Get covalent radius for an element"""
        # Simplified table of covalent radii in Angstroms
        radii = {
            1: 0.31,  # H
            6: 0.76,  # C
            7: 0.71,  # N
            8: 0.66,  # O
            # Add more elements as needed
        }
        return radii.get(atomic_number, 1.0)  # Default radius
    
    def _add_stretch_terms(self, b_matrix: np.ndarray, row: int, atoms: Tuple[int, int]):
        """Add B-matrix terms for bond stretch"""
        i, j = atoms
        rij = self.coords[j] - self.coords[i]
        dist = np.linalg.norm(rij)
        unit = rij / dist
        
        # Fill in B-matrix elements
        b_matrix[row, 3*i:3*i+3] = -unit
        b_matrix[row, 3*j:3*j+3] = unit
    
    def _add_bend_terms(self, b_matrix: np.ndarray, row: int, atoms: Tuple[int, int, int]):
        """Add B-matrix terms for angle bend"""
        i, j, k = atoms
        rji = self.coords[i] - self.coords[j]
        rjk = self.coords[k] - self.coords[j]
        
        dji = np.linalg.norm(rji)
        djk = np.linalg.norm(rjk)
        
        eji = rji / dji
        ejk = rjk / djk
        
        cos_theta = np.dot(eji, ejk)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        
        # Calculate partial derivatives
        for xyz in range(3):
            b_matrix[row, 3*i + xyz] = (eji[xyz] - cos_theta * ejk[xyz]) / (dji * sin_theta)
            b_matrix[row, 3*k + xyz] = (ejk[xyz] - cos_theta * eji[xyz]) / (djk * sin_theta)
            b_matrix[row, 3*j + xyz] = -(b_matrix[row, 3*i + xyz] + b_matrix[row, 3*k + xyz])

class RedundantInternalCoordinates:
    """Handles redundant internal coordinates and their transformations"""
    
    def __init__(self, coordinates: np.ndarray, atomic_numbers: np.ndarray, masses: np.ndarray):
        """
        Initialize redundant internal coordinate handler
        
        Args:
            coordinates: (N,3) array of Cartesian coordinates
            atomic_numbers: (N,) array of atomic numbers
            masses: (N,) array of atomic masses
        """
        self.coordinates = coordinates
        self.atomic_numbers = atomic_numbers
        self.masses = masses
        self.n_atoms = len(atomic_numbers)
        
    def calculate_redundant_coordinates(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray]:
        """
        Calculate redundant internal coordinates and B-matrix
        Based on bmatred Fortran subroutine
        
        Returns:
            Tuple containing:
            - B matrix (redundant)
            - B matrix (non-redundant)
            - List of internal coordinate definitions
            - Scale factors for each coordinate
        """
        # Get connectivity and determine internal coordinates
        connectivity = self._get_connectivity_matrix()
        internal_coords = self._determine_internal_coords(connectivity)
        
        # Initialize arrays
        n_redundant = len(internal_coords)
        n_cart = 3 * self.n_atoms
        br_matrix = np.zeros((n_redundant, n_cart))
        
        # Calculate values and derivatives for each coordinate
        z_values = []
        scale_factors = []
        
        for i, coord in enumerate(internal_coords):
            coord_type = coord['type']
            atoms = coord['atoms']
            
            # Calculate coordinate value and derivatives
            if coord_type == CoordinateType.STRETCH:
                value, derivs = self._calc_stretch(atoms)
                scale = 1.0
            elif coord_type == CoordinateType.BEND:
                value, derivs = self._calc_bend(atoms)
                scale = np.pi / 180.0  # Convert to degrees
            elif coord_type == CoordinateType.PROPER_TORSION:
                value, derivs = self._calc_torsion(atoms)
                scale = np.pi / 180.0
            elif coord_type == CoordinateType.IMPROPER_TORSION:
                value, derivs = self._calc_improper_torsion(atoms)
                scale = np.pi / 180.0
                
            # Store values
            z_values.append(value)
            scale_factors.append(scale)
            
            # Fill B-matrix row
            br_matrix[i] = derivs.flatten()
        
        # Convert redundant to non-redundant coordinates
        b_matrix, rank = self._redundant_to_nonredundant(br_matrix)
        
        return br_matrix, b_matrix, internal_coords, np.array(scale_factors)
    
    def _redundant_to_nonredundant(self, br_matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Convert redundant B-matrix to non-redundant form using SVD
        
        Args:
            br_matrix: Redundant B-matrix
            
        Returns:
            Tuple containing:
            - Non-redundant B-matrix
            - Rank of transformation
        """
        # Perform SVD
        u, s, vh = np.linalg.svd(br_matrix, full_matrices=False)
        
        # Determine rank using machine precision threshold
        eps = np.finfo(float).eps
        threshold = len(br_matrix) * np.amax(s) * eps
        rank = np.sum(s > threshold)
        
        # Construct non-redundant B-matrix
        u_nonred = u[:, :rank]
        s_nonred = s[:rank]
        vh_nonred = vh[:rank]
        
        b_matrix = np.dot(np.diag(s_nonred), vh_nonred)
        
        return b_matrix, rank
    
    def transform_coordinates(self, cartesian_disp: np.ndarray, 
                            b_matrix: np.ndarray) -> np.ndarray:
        """
        Transform Cartesian displacements to internal coordinates
        
        Args:
            cartesian_disp: Cartesian displacement vector
            b_matrix: B-matrix (redundant or non-redundant)
            
        Returns:
            Internal coordinate displacements
        """
        return np.dot(b_matrix, cartesian_disp)
    
    def transform_gradients(self, internal_grad: np.ndarray, 
                          b_matrix: np.ndarray) -> np.ndarray:
        """
        Transform internal coordinate gradients to Cartesian
        
        Args:
            internal_grad: Internal coordinate gradient vector
            b_matrix: B-matrix (redundant or non-redundant)
            
        Returns:
            Cartesian coordinate gradients
        """
        # Use pseudo-inverse for transformation
        b_inv = np.linalg.pinv(b_matrix)
        return np.dot(b_inv.T, internal_grad)
    
    def _calc_stretch(self, atoms: Tuple[int, int]) -> Tuple[float, np.ndarray]:
        """Calculate bond stretch value and derivatives"""
        i, j = atoms
        rij = self.coordinates[j] - self.coordinates[i]
        dist = np.linalg.norm(rij)
        unit = rij / dist
        
        derivs = np.zeros(3 * self.n_atoms)
        derivs[3*i:3*i+3] = -unit
        derivs[3*j:3*j+3] = unit
        
        return dist, derivs
    
    def _calc_bend(self, atoms: Tuple[int, int, int]) -> Tuple[float, np.ndarray]:
        """Calculate angle bend value and derivatives"""
        i, j, k = atoms
        rji = self.coordinates[i] - self.coordinates[j]
        rjk = self.coordinates[k] - self.coordinates[j]
        
        dji = np.linalg.norm(rji)
        djk = np.linalg.norm(rjk)
        
        eji = rji / dji
        ejk = rjk / djk
        
        cos_theta = np.dot(eji, ejk)
        if cos_theta > 1.0:
            cos_theta = 1.0
        elif cos_theta < -1.0:
            cos_theta = -1.0
            
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta)
        
        if abs(sin_theta) < 1e-10:
            # Handle linear case
            derivs = np.zeros(3 * self.n_atoms)
            return theta * 180.0 / np.pi, derivs
        
        derivs = np.zeros(3 * self.n_atoms)
        for xyz in range(3):
            derivs[3*i + xyz] = (eji[xyz] - cos_theta * ejk[xyz]) / (dji * sin_theta)
            derivs[3*k + xyz] = (ejk[xyz] - cos_theta * eji[xyz]) / (djk * sin_theta)
            derivs[3*j + xyz] = -(derivs[3*i + xyz] + derivs[3*k + xyz])
        
        return theta * 180.0 / np.pi, derivs

    def get_orthogonal_b_matrix(self, b_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get orthogonalized B-matrix and transformation matrices
        
        Args:
            b_matrix: Original B-matrix
            
        Returns:
            Tuple containing:
            - Orthogonalized B-matrix
            - Transformation matrix D
            - Normalization factors
        """
        orthog = BMatrixOrthogonalization()
        
        # First mass-weight if masses are available
        if hasattr(self, 'masses'):
            b_matrix = orthog.mass_weight_b_matrix(b_matrix, self.masses)
        
        # Perform orthogonalization
        bo_matrix, d_matrix, bnorm = orthog.orthogonalize_b_matrix(b_matrix)
        
        # Verify orthogonality
        is_orthog, overlap = orthog.verify_orthogonality(bo_matrix)
        if not is_orthog:
            warnings.warn("B-matrix orthogonalization may not be complete")
            
        return bo_matrix, d_matrix, bnorm

    def transform_with_orthogonal(self, cartesian_disp: np.ndarray, 
                                bo_matrix: np.ndarray, 
                                d_matrix: np.ndarray) -> np.ndarray:
        """
        Transform coordinates using orthogonalized B-matrix
        
        Args:
            cartesian_disp: Cartesian displacement vector
            bo_matrix: Orthogonalized B-matrix
            d_matrix: Transformation matrix
            
        Returns:
            Internal coordinate displacements
        """
        # First transform to orthogonal coordinates
        q_orthog = np.dot(bo_matrix, cartesian_disp)
        
        # Then transform back to original internal coordinates
        return np.dot(d_matrix, q_orthog)
