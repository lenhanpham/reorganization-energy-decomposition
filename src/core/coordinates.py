import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
from dataclasses import dataclass

class CoordinateType(Enum):
    STRETCH = 1
    BEND = 2
    PROPER_TORSION = 3
    IMPROPER_TORSION = 4

@dataclass
class InternalCoordinate:
    type: CoordinateType
    atoms: List[int]
    value: float
    scale: float

class MolecularCoordinates:
    def __init__(self, atomic_numbers: List[int], coordinates: np.ndarray):
        """
        Initialize molecular coordinates
        
        Args:
            atomic_numbers: List of atomic numbers
            coordinates: Nx3 array of cartesian coordinates in Angstroms
        """
        self.atomic_numbers = np.array(atomic_numbers)
        self.coordinates = np.array(coordinates)
        self.n_atoms = len(atomic_numbers)
        self.masses = self._get_atomic_masses()

    def _get_atomic_masses(self) -> np.ndarray:
        """Get atomic masses for all atoms"""
        # Implement atomic mass lookup
        pass

    def align(self, reference: 'MolecularCoordinates', 
              eigenvectors: Optional[np.ndarray] = None,
              force_reorder: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Align current molecular structure with a reference structure.
        Based on the Fortran align subroutine.
        
        Args:
            reference: Reference molecular structure to align with
            eigenvectors: Optional eigenvectors to transform (like force constants)
            force_reorder: Whether to force atom reordering (equivalent to iforder)
            
        Returns:
            Tuple containing:
            - Aligned coordinates
            - List of indices mapping current atoms to reference atoms
        """
        # Check that molecules have same number of atoms
        if self.n_atoms != reference.n_atoms:
            raise ValueError("Molecules must have same number of atoms")
            
        # Initialize variables
        coords = self.coordinates.copy()
        ref_coords = reference.coordinates
        n_atoms = self.n_atoms
        
        # Get atom type counts for both molecules
        atom_types = np.unique(self.atomic_numbers)
        type_counts = {at: np.sum(self.atomic_numbers == at) for at in atom_types}
        ref_type_counts = {at: np.sum(reference.atomic_numbers == at) for at in atom_types}
        
        if type_counts != ref_type_counts:
            raise ValueError("Molecules must have same atom composition")

        # Initialize atom mapping
        atom_mapping = list(range(n_atoms))
        
        if force_reorder:
            # Find optimal atom mapping by minimizing RMSD
            best_rmsd = float('inf')
            best_mapping = atom_mapping.copy()
            
            # Try different atom permutations (for same atom types)
            for at in atom_types:
                current_indices = np.where(self.atomic_numbers == at)[0]
                ref_indices = np.where(reference.atomic_numbers == at)[0]
                
                if len(current_indices) > 1:
                    from itertools import permutations
                    for perm in permutations(current_indices):
                        # Create test mapping
                        test_mapping = atom_mapping.copy()
                        for i, idx in enumerate(current_indices):
                            test_mapping[idx] = ref_indices[i]
                            
                        # Calculate RMSD for this mapping
                        test_coords = coords[test_mapping]
                        rmsd = self._calculate_rmsd(test_coords, ref_coords)
                        
                        if rmsd < best_rmsd:
                            best_rmsd = rmsd
                            best_mapping = test_mapping.copy()
            
            atom_mapping = best_mapping
            coords = coords[atom_mapping]

        # Calculate optimal rotation matrix using Kabsch algorithm
        rotation_matrix = self._get_optimal_rotation(coords, ref_coords)
        
        # Apply rotation to coordinates
        aligned_coords = np.dot(coords, rotation_matrix.T)
        
        # Transform eigenvectors if provided
        if eigenvectors is not None:
            n_coords = 3 * n_atoms
            transform = np.kron(rotation_matrix, np.eye(n_atoms))
            eigenvectors = np.dot(transform.T, eigenvectors)
            
        return aligned_coords, atom_mapping

    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate Root Mean Square Deviation between two coordinate sets"""
        return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

    def _get_optimal_rotation(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """
        Calculate optimal rotation matrix using Kabsch algorithm
        """
        # Center both coordinate sets
        com1 = np.mean(coords1, axis=0)
        com2 = np.mean(coords2, axis=0)
        coords1_centered = coords1 - com1
        coords2_centered = coords2 - com2
        
        # Calculate covariance matrix
        covariance = np.dot(coords1_centered.T, coords2_centered)
        
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(covariance)
        
        # Ensure right-handed coordinate system
        d = np.linalg.det(np.dot(Vt.T, U.T))
        if d < 0:
            Vt[-1] *= -1
            
        # Calculate rotation matrix
        rotation = np.dot(Vt.T, U.T)
        
        return rotation

    def get_internal_coordinates(self) -> np.ndarray:
        """Convert cartesian to internal coordinates"""
        # Implement Z-matrix or other internal coordinate conversion
        pass

    def get_cartesian_coordinates(self) -> np.ndarray:
        """Convert internal to cartesian coordinates"""
        pass

    def calculate_inertia_tensor(self, coords: Optional[np.ndarray] = None, 
                               masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate moment of inertia tensor using real atomic masses
        Replaces inertm subroutine's inertia tensor calculation
        
        Args:
            coords: Optional coordinate array to use instead of self.coordinates
            masses: Optional masses array to use instead of self.masses
        
        Returns:
            3x3 inertia tensor matrix
        """
        if coords is None:
            coords = self.coordinates
        if masses is None:
            masses = self.masses
        
        # Initialize inertia tensor
        inertia = np.zeros((3, 3))
        
        # Calculate center of mass
        total_mass = np.sum(masses)
        com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
        
        # Center coordinates
        coords_centered = coords - com
        
        # Build inertia tensor
        for i in range(len(masses)):
            # Diagonal elements
            inertia[0,0] += masses[i] * (coords_centered[i,1]**2 + coords_centered[i,2]**2)
            inertia[1,1] += masses[i] * (coords_centered[i,0]**2 + coords_centered[i,2]**2)
            inertia[2,2] += masses[i] * (coords_centered[i,0]**2 + coords_centered[i,1]**2)
            
            # Off-diagonal elements
            inertia[0,1] -= masses[i] * coords_centered[i,0] * coords_centered[i,1]
            inertia[0,2] -= masses[i] * coords_centered[i,0] * coords_centered[i,2]
            inertia[1,2] -= masses[i] * coords_centered[i,1] * coords_centered[i,2]
        
        # Symmetrize
        inertia[1,0] = inertia[0,1]
        inertia[2,0] = inertia[0,2]
        inertia[2,1] = inertia[1,2]
        
        return inertia

    def inertm(self, eigenvectors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform molecule to inertial coordinate system using real atomic masses.
        Based on the Fortran inertm subroutine.
        
        Args:
            eigenvectors: Optional eigenvectors to transform along with coordinates
                (equivalent to c matrix in Fortran version)
                
        Returns:
            Tuple containing:
            - Transformed coordinates
            - Rotation matrix used for transformation
        """
        # Calculate inertia tensor
        inertia = self.calculate_inertia_tensor()
        
        # Diagonalize inertia tensor
        eigenvals, eigenvecs = np.linalg.eigh(inertia)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        rotation = eigenvecs[:, idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1
        
        # Center coordinates at center of mass
        total_mass = np.sum(self.masses)
        com = np.sum(self.coordinates * self.masses[:, np.newaxis], axis=0) / total_mass
        coords_centered = self.coordinates - com
        
        # Apply rotation to coordinates
        new_coords = np.dot(coords_centered, rotation)
        
        # Transform eigenvectors if provided
        if eigenvectors is not None:
            n_atoms = len(self.atomic_numbers)
            # Create block diagonal rotation matrix for all coordinates
            full_rotation = np.kron(rotation, np.eye(n_atoms))
            # Transform eigenvectors
            eigenvectors = np.dot(eigenvectors, full_rotation)
        
        # Update internal coordinates
        self.coordinates = new_coords + com
        
        # Print diagnostic information (similar to Fortran version)
        if hasattr(self, 'logger'):
            self.logger.info("Full Moment of inertia matrix:")
            for i in range(3):
                self.logger.info(f"{i+1} {' '.join(f'{x:10.4f}' for x in inertia[i,:i+1])}")
            
            self.logger.info(f"Principal moments: {' '.join(f'{x:10.4f}' for x in eigenvals)}")
            for i in range(3):
                self.logger.info(f"{i+1} {' '.join(f'{x:10.5f}' for x in rotation[i])}")
            
            self.logger.info("Coordinates after full-mass inertial transform:")
            for i in range(len(self.atomic_numbers)):
                coord_str = ' '.join(f'{x:10.6f}' for x in self.coordinates[i])
                self.logger.info(f"{i+1} {coord_str}")
        
        return self.coordinates, rotation

    def rotate_to_standard_orientation(self) -> None:
        """
        Rotate molecule to standard orientation
        Based on original orient subroutine
        """
        # Implementation here
        pass

    def calculate_b_matrix(self, nosym: int = 1) -> Tuple[np.ndarray, np.ndarray, List[InternalCoordinate]]:
        """
        Calculate Wilson B-matrix and its inverse for internal coordinate transformations.
        Based on the Fortran bmat subroutine.
        
        Args:
            nosym: Symmetry control parameter
                  -1 -> all variables unique (no symmetry)
                   0 -> molecule is planar (no torsional variables)
                   1 -> use full point-group symmetry
        
        Returns:
            Tuple containing:
            - B matrix
            - B inverse matrix
            - List of internal coordinates
        """
        n_atoms = self.n_atoms
        coords = self.coordinates
        
        # Initialize connectivity matrix
        connectivity = self._get_connectivity_matrix()
        
        # Get internal coordinates
        internal_coords = self._determine_internal_coordinates(connectivity, nosym)
        n_internals = len(internal_coords)
        
        # Initialize B matrix
        b_matrix = np.zeros((n_internals, 3 * n_atoms))
        
        # Fill B matrix
        for i, coord in enumerate(internal_coords):
            if coord.type == CoordinateType.STRETCH:
                self._add_stretch_terms(b_matrix, i, coord.atoms, coords)
            elif coord.type == CoordinateType.BEND:
                self._add_bend_terms(b_matrix, i, coord.atoms, coords)
            elif coord.type == CoordinateType.PROPER_TORSION:
                self._add_torsion_terms(b_matrix, i, coord.atoms, coords)
            elif coord.type == CoordinateType.IMPROPER_TORSION:
                self._add_improper_terms(b_matrix, i, coord.atoms, coords)
        
        # Calculate B inverse
        b_inv = np.linalg.pinv(b_matrix)
        
        return b_matrix, b_inv, internal_coords

    def _get_connectivity_matrix(self) -> np.ndarray:
        """Calculate molecular connectivity based on atomic distances"""
        coords = self.coordinates
        n_atoms = self.n_atoms
        connectivity = np.zeros((n_atoms, n_atoms), dtype=bool)
        
        # Covalent radii (in Angstroms) - simplified version
        covalent_radii = {
            1: 0.31,  # H
            6: 0.76,  # C
            7: 0.71,  # N
            8: 0.66,  # O
            # Add more elements as needed
        }
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                r1 = covalent_radii.get(self.atomic_numbers[i], 1.0)
                r2 = covalent_radii.get(self.atomic_numbers[j], 1.0)
                if dist < (r1 + r2) * 1.3:  # 1.3 is a tolerance factor
                    connectivity[i,j] = connectivity[j,i] = True
                    
        return connectivity

    def _determine_internal_coordinates(self, 
                                     connectivity: np.ndarray, 
                                     nosym: int) -> List[InternalCoordinate]:
        """Determine internal coordinates based on molecular connectivity"""
        coords = []
        n_atoms = self.n_atoms
        
        # Add bond stretches
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if connectivity[i,j]:
                    coords.append(InternalCoordinate(
                        type=CoordinateType.STRETCH,
                        atoms=[i, j],
                        value=np.linalg.norm(self.coordinates[i] - self.coordinates[j]),
                        scale=1.0
                    ))
        
        # Add bond angles
        for i in range(n_atoms):
            bonded_to_i = np.where(connectivity[i])[0]
            for j in bonded_to_i:
                for k in bonded_to_i:
                    if j < k:
                        coords.append(InternalCoordinate(
                            type=CoordinateType.BEND,
                            atoms=[j, i, k],
                            value=self._calculate_angle(j, i, k),
                            scale=np.pi/180.0  # convert to radians
                        ))
        
        # Add torsions if not planar
        if nosym != 0:
            torsions = self._find_torsions(connectivity)
            for t in torsions:
                coords.append(InternalCoordinate(
                    type=CoordinateType.PROPER_TORSION,
                    atoms=t,
                    value=self._calculate_torsion(t),
                    scale=np.pi/180.0
                ))
        
        return coords

    def _add_stretch_terms(self, 
                         b_matrix: np.ndarray, 
                         row: int, 
                         atoms: List[int], 
                         coords: np.ndarray) -> None:
        """Add B-matrix terms for bond stretch"""
        i, j = atoms
        rij = coords[j] - coords[i]
        dist = np.linalg.norm(rij)
        unit = rij / dist
        
        for k in range(3):
            b_matrix[row, 3*i + k] = -unit[k]
            b_matrix[row, 3*j + k] = unit[k]

    def _add_bend_terms(self, 
                       b_matrix: np.ndarray, 
                       row: int, 
                       atoms: List[int], 
                       coords: np.ndarray) -> None:
        """Add B-matrix terms for bond angle"""
        i, j, k = atoms
        rji = coords[i] - coords[j]
        rjk = coords[k] - coords[j]
        
        dji = np.linalg.norm(rji)
        djk = np.linalg.norm(rjk)
        
        eji = rji / dji
        ejk = rjk / djk
        
        cos_theta = np.dot(eji, ejk)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        
        if sin_theta < 1e-10:
            return  # Avoid division by zero for linear angles
            
        for l in range(3):
            b_matrix[row, 3*i + l] = (cos_theta * eji[l] - ejk[l]) / (dji * sin_theta)
            b_matrix[row, 3*k + l] = (cos_theta * ejk[l] - eji[l]) / (djk * sin_theta)
            b_matrix[row, 3*j + l] = -(b_matrix[row, 3*i + l] + b_matrix[row, 3*k + l])

    def _calculate_angle(self, i: int, j: int, k: int) -> float:
        """Calculate angle between three atoms in degrees"""
        rji = self.coordinates[i] - self.coordinates[j]
        rjk = self.coordinates[k] - self.coordinates[j]
        
        cos_theta = np.dot(rji, rjk) / (np.linalg.norm(rji) * np.linalg.norm(rjk))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta) * 180.0 / np.pi

    def _find_torsions(self, connectivity: np.ndarray) -> List[List[int]]:
        """Find proper torsion angles in molecule"""
        torsions = []
        n_atoms = self.n_atoms
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if not connectivity[i,j]:
                    continue
                for k in range(n_atoms):
                    if not connectivity[j,k] or k == i:
                        continue
                    for l in range(n_atoms):
                        if not connectivity[k,l] or l == j or l == i:
                            continue
                        torsions.append([i, j, k, l])
        
        return torsions

    def _calculate_torsion(self, atoms: List[int]) -> float:
        """Calculate torsion angle between four atoms in degrees"""
        i, j, k, l = atoms
        rji = self.coordinates[i] - self.coordinates[j]
        rjk = self.coordinates[k] - self.coordinates[j]
        rkl = self.coordinates[l] - self.coordinates[k]
        
        # Calculate normal vectors to the planes
        n1 = np.cross(rji, rjk)
        n2 = np.cross(rjk, rkl)
        
        # Calculate torsion angle
        cos_phi = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        
        # Determine sign of torsion
        sign = np.sign(np.dot(np.cross(n1, n2), rjk))
        
        return sign * np.arccos(cos_phi) * 180.0 / np.pi

    def _bmat1(self,
              coords: np.ndarray,
              masses: np.ndarray,
              atomic_numbers: np.ndarray,
              nosym: int,
              connectivity: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, List[InternalCoordinate], np.ndarray]:
        """
        Calculate B-matrix and internal coordinates.
        Based on the Fortran bmat1 subroutine.

        Args:
            coords: (N,3) array of Cartesian coordinates
            masses: (N,) array of atomic masses
            atomic_numbers: (N,) array of atomic numbers
            nosym: Symmetry control parameter:
                  -1 -> all variables unique (no symmetry)
                   0 -> molecule is planar (no torsional variables)
                   1 -> use full point-group symmetry
            connectivity: Optional pre-computed connectivity matrix

        Returns:
            Tuple containing:
            - B matrix
            - B inverse matrix
            - List of internal coordinates
            - Scale factors for each coordinate
        """
        n_atoms = len(atomic_numbers)
        n3 = 3 * n_atoms

        # Get or compute connectivity matrix
        if connectivity is None:
            connectivity = self._get_connectivity_matrix()

        # Initialize arrays
        z_vars = []  # Will store internal coordinate values
        scales = []  # Will store scaling factors
        coord_types = []  # Will store coordinate types
        coord_atoms = []  # Will store atoms involved in each coordinate
        
        # Center of mass calculation
        total_mass = np.sum(masses)
        com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
        coords_centered = coords - com

        # Get principal axes
        inertia = self._calculate_inertia_tensor(coords_centered, masses)
        _, axes = np.linalg.eigh(inertia)

        # Initialize B-matrix with translations and rotations
        n_coords = n3  # Start with all degrees of freedom
        b_matrix = np.zeros((n_coords, n3))

        # Add translational coordinates (first 3 rows)
        for i in range(3):
            for j in range(n_atoms):
                b_matrix[i, 3*j:3*j+3] = np.eye(3)[i] * np.sqrt(masses[j])
                
        # Add rotational coordinates (next 3 rows)
        for i in range(3):
            for j in range(n_atoms):
                r = coords_centered[j]
                L = self._cross_matrix(r)
                b_matrix[i+3, 3*j:3*j+3] = L[i] * np.sqrt(masses[j])

        # Process stretches
        stretch_idx = 6  # Start after translations and rotations
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if connectivity[i,j]:
                    rij = coords[j] - coords[i]
                    dist = np.linalg.norm(rij)
                    unit = rij / dist
                    
                    # Add to B-matrix
                    for k in range(3):
                        b_matrix[stretch_idx, 3*i + k] = -unit[k]
                        b_matrix[stretch_idx, 3*j + k] = unit[k]
                    
                    # Store coordinate information
                    z_vars.append(dist)
                    scales.append(1.0)  # Stretches use unit scaling
                    coord_types.append(CoordinateType.STRETCH)
                    coord_atoms.append([i, j])
                    stretch_idx += 1

        # Process angles
        for i in range(n_atoms):
            bonded = np.where(connectivity[i])[0]
            for j in bonded:
                for k in bonded:
                    if j < k:
                        rji = coords[j] - coords[i]
                        rki = coords[k] - coords[i]
                        
                        dji = np.linalg.norm(rji)
                        dki = np.linalg.norm(rki)
                        
                        cos_theta = np.dot(rji, rki) / (dji * dki)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        theta = np.arccos(cos_theta)
                        
                        # Add to B-matrix
                        self._add_angle_terms(b_matrix, stretch_idx, 
                                            [j, i, k], coords)
                        
                        # Store coordinate information
                        z_vars.append(theta * 180.0 / np.pi)
                        scales.append(np.pi / 180.0)
                        coord_types.append(CoordinateType.BEND)
                        coord_atoms.append([j, i, k])
                        stretch_idx += 1

        # Process torsions if not planar
        if nosym != 0:
            torsions = self._find_torsions(connectivity)
            for t in torsions:
                i, j, k, l = t
                phi = self._calculate_torsion_angle(coords[i], coords[j], 
                                                  coords[k], coords[l])
                
                # Add to B-matrix
                self._add_torsion_terms(b_matrix, stretch_idx, t, coords)
                
                # Store coordinate information
                z_vars.append(phi)
                scales.append(np.pi / 180.0)
                coord_types.append(CoordinateType.PROPER_TORSION)
                coord_atoms.append(t)
                stretch_idx += 1

        # Create internal coordinates list
        internal_coords = [
            InternalCoordinate(type=t, atoms=a, value=v, scale=s)
            for t, a, v, s in zip(coord_types, coord_atoms, z_vars, scales)
        ]

        # Calculate B inverse using pseudo-inverse
        b_inv = np.linalg.pinv(b_matrix)

        return b_matrix, b_inv, internal_coords, np.array(scales)

    def _cross_matrix(self, r: np.ndarray) -> np.ndarray:
        """
        Create cross product matrix for angular momentum calculation
        """
        return np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])

    def _calculate_torsion_angle(self, 
                               r1: np.ndarray, 
                               r2: np.ndarray, 
                               r3: np.ndarray, 
                               r4: np.ndarray) -> float:
        """
        Calculate torsion angle between four points in degrees
        """
        b1 = r2 - r1
        b2 = r3 - r2
        b3 = r4 - r3

        # Normal vectors to planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        # Normalize vectors
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)

        # Calculate angle
        cos_phi = np.dot(n1, n2)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)

        # Determine sign
        sign = np.sign(np.dot(np.cross(n1, n2), b2))
        phi = sign * np.arccos(cos_phi)

        return phi * 180.0 / np.pi

    def _add_angle_terms(self,
                        b_matrix: np.ndarray,
                        row: int,
                        atoms: List[int],
                        coords: np.ndarray) -> None:
        """
        Add B-matrix terms for bond angle
        """
        i, j, k = atoms
        rji = coords[i] - coords[j]
        rjk = coords[k] - coords[j]
        
        dji = np.linalg.norm(rji)
        djk = np.linalg.norm(rjk)
        
        eji = rji / dji
        ejk = rjk / djk
        
        cos_theta = np.dot(eji, ejk)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        
        if sin_theta < 1e-10:
            return  # Skip nearly linear angles
            
        for l in range(3):
            b_matrix[row, 3*i + l] = (cos_theta * eji[l] - ejk[l]) / (dji * sin_theta)
            b_matrix[row, 3*k + l] = (cos_theta * ejk[l] - eji[l]) / (djk * sin_theta)
            b_matrix[row, 3*j + l] = -(b_matrix[row, 3*i + l] + b_matrix[row, 3*k + l])

    def _add_torsion_terms(self,
                          b_matrix: np.ndarray,
                          row: int,
                          atoms: List[int],
                          coords: np.ndarray) -> None:
        """
        Add B-matrix terms for torsion angle
        """
        i, j, k, l = atoms
        rji = coords[i] - coords[j]
        rjk = coords[k] - coords[j]
        rkl = coords[l] - coords[k]
        
        # Calculate cross products and norms
        n1 = np.cross(rji, rjk)
        n2 = np.cross(rjk, rkl)
        
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        rjk_norm = np.linalg.norm(rjk)
        
        if n1_norm < 1e-10 or n2_norm < 1e-10 or rjk_norm < 1e-10:
            return  # Skip problematic torsions
        
        # Calculate derivatives
        for m in range(3):
            # Contribution from first atom
            b_matrix[row, 3*i + m] = n1[m] / n1_norm
            
            # Contribution from last atom
            b_matrix[row, 3*l + m] = n2[m] / n2_norm
            
            # Contributions from middle atoms (more complex)
            # These terms need careful derivation from the torsion angle formula
            b_matrix[row, 3*j + m] = -n1[m] / n1_norm - \
                                    (np.dot(rji, rjk) / rjk_norm**2) * n2[m] / n2_norm
            
            b_matrix[row, 3*k + m] = -n2[m] / n2_norm + \
                                    (np.dot(rkl, rjk) / rjk_norm**2) * n1[m] / n1_norm

    def getrtr(self, coords: np.ndarray, ref_coords: np.ndarray) -> np.ndarray:
        """
        Find rotation matrix which moves vectors in coords to align with ref_coords.
        Based on the Fortran getrtr subroutine.
        
        Args:
            coords: Current coordinate set (Nx3 array)
            ref_coords: Reference coordinate set (Nx3 array)
        
        Returns:
            3x3 rotation matrix
        """
        # Get first three atoms to define coordinate system
        x = coords[:3]
        y = ref_coords[:3]
        
        # Calculate vectors defining local coordinate systems
        xv = np.zeros((3, 3))
        yv = np.zeros((3, 3))
        
        # First two vectors from differences between points
        xv[:, 0] = x[1] - x[0]  # Vector from atom 1 to 2
        xv[:, 1] = x[2] - x[0]  # Vector from atom 1 to 3
        yv[:, 0] = y[1] - y[0]  # Vector from atom 1 to 2 (reference)
        yv[:, 1] = y[2] - y[0]  # Vector from atom 1 to 3 (reference)
        
        # Third vector is cross product of first two vectors
        xv[:, 2] = np.cross(xv[:, 0], xv[:, 1])
        yv[:, 2] = np.cross(yv[:, 0], yv[:, 1])
        
        # Calculate rotation matrix: R = Y * X^(-1)
        # Using the fact that for orthogonal matrices, inverse = transpose
        try:
            # Try direct matrix inversion first
            xv_inv = np.linalg.inv(xv)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            xv_inv = np.linalg.pinv(xv)
        
        # Calculate rotation matrix
        rotation = np.dot(yv, xv_inv)
        
        # Ensure rotation matrix is orthogonal
        u, _, vh = np.linalg.svd(rotation)
        rotation = np.dot(u, vh)
        
        # Ensure right-handed coordinate system
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1
        
        return rotation

    def apply_rotation(self, coords: np.ndarray, rotation: np.ndarray, 
                      eigenvectors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply rotation matrix to coordinates and optionally to eigenvectors.
        
        Args:
            coords: Coordinates to rotate (Nx3 array)
            rotation: 3x3 rotation matrix
            eigenvectors: Optional eigenvectors to transform
        
        Returns:
            Tuple of:
            - Rotated coordinates
            - Transformed eigenvectors (if provided)
        """
        # Apply rotation to coordinates
        rotated_coords = np.dot(coords, rotation.T)
        
        # Transform eigenvectors if provided
        transformed_evecs = None
        if eigenvectors is not None:
            n_atoms = len(coords)
            # Create block diagonal rotation matrix for all coordinates
            full_rotation = np.kron(rotation, np.eye(n_atoms))
            # Transform eigenvectors
            transformed_evecs = np.dot(full_rotation.T, eigenvectors)
        
        return rotated_coords, transformed_evecs

    def align_to_reference(self, ref_coords: np.ndarray, 
                          eigenvectors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align current coordinates to reference coordinates using getrtr.
        
        Args:
            ref_coords: Reference coordinates to align to
            eigenvectors: Optional eigenvectors to transform
        
        Returns:
            Tuple containing:
            - Aligned coordinates
            - Rotation matrix used
        """
        # Get rotation matrix
        rotation = self.getrtr(self.coordinates, ref_coords)
        
        # Apply rotation
        aligned_coords, transformed_evecs = self.apply_rotation(
            self.coordinates, rotation, eigenvectors
        )
        
        # Update internal coordinates
        self.coordinates = aligned_coords
        
        # Update eigenvectors if provided
        if eigenvectors is not None:
            return aligned_coords, rotation, transformed_evecs
        
        return aligned_coords, rotation
