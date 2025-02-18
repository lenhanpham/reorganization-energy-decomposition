import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class OutputWriter:
    """Handles writing of supplementary output files"""
    
    def __init__(self):
        self.au_to_cm = 219474.63
        self.au_to_ang = 0.529177249
        
    def write_supplementary_data(self,
                               filename: str,
                               molecule_name: str,
                               point_group: str,
                               coords: np.ndarray,
                               atomic_numbers: np.ndarray,
                               frequencies: np.ndarray,
                               normal_modes: np.ndarray,
                               ir_intensities: Optional[np.ndarray] = None,
                               zpt_lengths: Optional[np.ndarray] = None) -> None:
        """
        Write supplementary output data
        
        Args:
            filename: Output file name
            molecule_name: Name of molecule
            point_group: Point group symbol
            coords: Cartesian coordinates
            atomic_numbers: Atomic numbers
            frequencies: Vibrational frequencies
            normal_modes: Normal mode vectors
            ir_intensities: Optional IR intensities
            zpt_lengths: Optional zero-point lengths
        """
        with open(filename, 'w') as f:
            # Write header
            f.write(f"{'-'*78}\n")
            f.write(f"Results for: {molecule_name} {point_group}\n")
            f.write(f"{'-'*78}\n\n")
            
            # Write geometry
            f.write("Geometry (Angstroms):\n")
            for i, (num, coord) in enumerate(zip(atomic_numbers, coords)):
                f.write(f"{self._atomic_symbol(num):2s} {coord[0]:12.6f} "
                       f"{coord[1]:12.6f} {coord[2]:12.6f}\n")
            f.write("\n")
            
            # Write vibrational data in blocks of 6
            n_modes = len(frequencies)
            for i in range(0, n_modes, 6):
                block = slice(i, min(i+6, n_modes))
                
                # Mode numbers
                f.write("Mode   " + "".join(f"{j+1:10d}" for j in range(i, min(i+6, n_modes))) + "\n")
                
                # Frequencies
                f.write("Freq   " + "".join(f"{freq:10.2f}" for freq in frequencies[block]) + "\n")
                
                # IR intensities
                if ir_intensities is not None:
                    f.write("IR Int " + "".join(f"{ir:10.2f}" for ir in ir_intensities[block]) + "\n")
                
                # Zero-point lengths
                if zpt_lengths is not None:
                    f.write("ZPT    " + "".join(f"{zpt:10.5f}" for zpt in zpt_lengths[block]) + "\n")
                
                f.write("\n")
                
                # Normal modes
                for j in range(len(coords)):
                    for k in range(3):
                        atom_sym = self._atomic_symbol(atomic_numbers[j])
                        f.write(f"{atom_sym}{j+1}{['x','y','z'][k]} ")
                        f.write("".join(f"{normal_modes[3*j+k,m]:10.5f}" 
                               for m in range(i, min(i+6, n_modes))) + "\n")
                f.write("\n")
    
    @staticmethod
    def _atomic_symbol(atomic_number: int) -> str:
        """Get atomic symbol from atomic number"""
        symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']  # Add more as needed
        return symbols[atomic_number-1] if atomic_number <= len(symbols) else str(atomic_number)