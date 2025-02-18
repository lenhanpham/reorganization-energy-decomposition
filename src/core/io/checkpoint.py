import numpy as np
from typing import Dict, Optional, Tuple

class CheckpointReader:
    """Handles reading of formatted checkpoint files"""
    
    def read_fchk(self, filename: str) -> Dict:
        """
        Read Gaussian formatted checkpoint file
        
        Args:
            filename: Path to .fchk file
            
        Returns:
            Dictionary containing checkpoint data
        """
        data = {}
        current_section = None
        array_data = []
        
        with open(filename, 'r') as f:
            for line in f:
                if 'Number of atoms' in line:
                    data['n_atoms'] = int(line.split()[-1])
                elif 'Cartesian Force Constants' in line:
                    current_section = 'force_constants'
                    array_size = int(line.split()[-1])
                    array_data = []
                elif current_section == 'force_constants':
                    values = [float(x) for x in line.split()]
                    array_data.extend(values)
                    if len(array_data) >= array_size:
                        data['force_constants'] = self._convert_lower_triangle(
                            array_data, data['n_atoms']*3)
                        current_section = None
                        
        return data
    
    def _convert_lower_triangle(self, 
                              triangle: list, 
                              matrix_size: int) -> np.ndarray:
        """
        Convert lower triangle to full matrix
        
        Args:
            triangle: Lower triangle elements
            matrix_size: Size of square matrix
            
        Returns:
            Full symmetric matrix
        """
        matrix = np.zeros((matrix_size, matrix_size))
        idx = 0
        for i in range(matrix_size):
            for j in range(i+1):
                matrix[i,j] = triangle[idx]
                matrix[j,i] = triangle[idx]
                idx += 1
        return matrix