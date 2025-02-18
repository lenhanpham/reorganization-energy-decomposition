import numpy as np
from typing import Tuple, Optional, List

class ForceConstants:
    """Handles force constant calculations and transformations"""
    
    def __init__(self):
        self.au_to_cm = 219474.63  # Conversion factor: atomic units to cm⁻¹
    
    def read_force_constants(self, 
                           filename: str,
                           n_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read force constants from various quantum chemistry outputs
        
        Args:
            filename: Input file name
            n_atoms: Number of atoms
            
        Returns:
            Tuple containing:
            - Force constant matrix
            - Frequencies (if available)
        """
        # Determine file type and read accordingly
        if filename.endswith('.fchk'):
            return self._read_gaussian_fchk(filename, n_atoms)
        elif filename.endswith('.out'):
            return self._read_output_file(filename, n_atoms)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    def mass_weight_force_constants(self,
                                  fc: np.ndarray,
                                  masses: np.ndarray) -> np.ndarray:
        """
        Apply mass-weighting to force constant matrix
        
        Args:
            fc: Force constant matrix
            masses: Atomic masses
            
        Returns:
            Mass-weighted force constant matrix
        """
        # Repeat masses for x,y,z components
        mass_weights = np.repeat(np.sqrt(masses), 3)
        return fc * np.outer(mass_weights, mass_weights)
    
    def diagonalize_force_constants(self,
                                  fc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize force constant matrix
        
        Args:
            fc: Force constant matrix
            
        Returns:
            Tuple containing:
            - Eigenvalues (frequencies squared)
            - Eigenvectors (normal modes)
        """
        # Ensure matrix is symmetric
        fc = (fc + fc.T) / 2
        
        # Diagonalize
        eigenvals, eigenvecs = np.linalg.eigh(fc)
        
        return eigenvals, eigenvecs
