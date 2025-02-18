from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SupplementaryData:
    """Container for supplementary data parameters"""
    name: str
    point_group: str
    energy: float
    coordinates: np.ndarray
    atomic_numbers: np.ndarray
    symmetry_labels: List[str]
    frequencies: np.ndarray
    ir_intensities: Optional[np.ndarray]
    zero_point_lengths: np.ndarray
    normal_modes: np.ndarray
    internal_coords: Optional[np.ndarray] = None
    coord_types: Optional[np.ndarray] = None
    coord_definitions: Optional[np.ndarray] = None
    coord_scales: Optional[np.ndarray] = None

class SupplementaryWriter:
    """Handles writing supplementary data files"""
    
    def __init__(self):
        self.atomic_symbols = {
            1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
            15: 'P', 16: 'S', 17: 'Cl'
            # Add more as needed
        }
        self.read_count = 0
        
    def write_supplementary_data(self, 
                               data: SupplementaryData,
                               output_dir: Path,
                               freq_threshold: float = 1e-6) -> None:
        """
        Write supplementary data files (equivalent to wsupp subroutine)
        
        Args:
            data: SupplementaryData object containing all required data
            output_dir: Directory for output files
            freq_threshold: Threshold for considering frequencies as zero
        """
        # Increment read counter
        self.read_count += 1
        
        # Determine calculation type
        opt_type = 'full'
        if 'CI' in data.name or 'TS' in data.name:
            opt_type = 'part'
            
        # Count non-zero frequencies
        non_zero_modes = np.where(np.abs(data.frequencies) > freq_threshold)[0]
        n_nonzero = len(non_zero_modes)
        
        # Determine frequency calculation completeness
        freq_label = 'none'
        if n_nonzero > 0:
            freq_label = 'some'
        if n_nonzero == 3 * len(data.atomic_numbers) - 6:
            freq_label = 'full'
            
        # Write to supplem.ind
        with open(output_dir / 'supplem.ind', 'a') as f:
            f.write(f"{self.read_count:3d}  {data.name:<30s} "
                   f"{data.point_group:<4s} {opt_type:<4s} {freq_label:<4s}\n")
            
        # Write to supplem.dat
        with open(output_dir / 'supplem.dat', 'a') as f:
            # Write header
            f.write('\n' + '_' * 78 + '\n')
            f.write(f"{'Results for: ':>20s}{data.name} {data.point_group}\n")
            f.write('_' * 20 + '\n\n')
            
            # Write energy and coordinates
            f.write(f"SCF Done: E = {data.energy:20.12f}\n\n")
            for i, (atom, coord) in enumerate(zip(data.atomic_numbers, data.coordinates)):
                symbol = self.atomic_symbols.get(atom, 'X')
                coord_str = ' '.join(f'{x:12.6f}' for x in coord)
                f.write(f"{symbol:<4s} {coord_str}\n")
            
            # Write vibrational data in blocks of 9
            for i in range(0, n_nonzero, 9):
                block = slice(i, min(i + 9, n_nonzero))
                modes = non_zero_modes[block]
                
                # Mode numbers
                f.write('\nMode   ' + ' '.join(f'{m+1:8d}' for m in modes))
                
                # Symmetry labels
                f.write('\nSymm   ' + ' '.join(f'{data.symmetry_labels[m]:>8s}' 
                                              for m in modes))
                
                # Frequencies
                f.write('\nFreq   ' + ' '.join(f'{data.frequencies[m]:8.2f}' 
                                              for m in modes))
                
                # IR intensities if available
                if data.ir_intensities is not None:
                    f.write('\nIR Int ' + ' '.join(f'{data.ir_intensities[m]:8.2f}' 
                                                  for m in modes))
                
                # Zero-point lengths
                f.write('\nZpt l  ' + ' '.join(f'{data.zero_point_lengths[m]:8.5f}' 
                                              for m in modes))
                
                # Normal modes
                f.write('\n')
                for j in range(len(data.atomic_numbers)):
                    for k, xyz in enumerate(['x', 'y', 'z']):
                        idx = 3 * j + k
                        symbol = self.atomic_symbols.get(data.atomic_numbers[j], 'X')
                        mode_str = ' '.join(f'{data.normal_modes[idx, m]:8.4f}' 
                                          for m in modes)
                        f.write(f"{symbol}{j+1}{xyz}  {mode_str}\n")
                
        # Write internal coordinates if available
        if (data.internal_coords is not None and 
            data.coord_types is not None and 
            data.coord_definitions is not None):
            
            with open(output_dir / 'supplem.geom', 'a') as f:
                f.write(f"{data.name} {data.point_group}\n")
                f.write(f"{len(data.internal_coords):5d}\n")
                
                for i, (z, typ, defn, scale) in enumerate(zip(
                    data.internal_coords, 
                    data.coord_types,
                    data.coord_definitions,
                    data.coord_scales
                )):
                    defn_str = ' '.join(f'{d:4d}' for d in defn)
                    f.write(f"{i+1:5d}{typ:2d} {defn_str} {z*scale:11.5f}\n")