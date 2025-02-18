import numpy as np
from typing import Tuple, Optional

class MatrixOperations:
    """Handles matrix operations from Fortran routines"""
    
    def tred2e(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Householder reduction of a real symmetric matrix to tridiagonal form.
        Python implementation of the Fortran tred2e subroutine.
        
        Args:
            matrix: Square symmetric matrix to reduce
            
        Returns:
            Tuple containing:
            - Diagonal elements
            - Off-diagonal elements
            - Transformation matrix
        """
        n = matrix.shape[0]
        
        # Make a copy to avoid modifying input
        a = matrix.copy()
        
        # Initialize output arrays
        d = np.zeros(n)  # Diagonal elements
        e = np.zeros(n)  # Off-diagonal elements
        z = np.eye(n)    # Transformation matrix
        
        # Working arrays
        work = np.zeros(n)
        
        for i in range(n-1, 0, -1):
            l = i - 1
            h = 0.0
            scale = 0.0
            
            # Scale row (if needed)
            if l > 0:
                scale = np.sum(np.abs(a[i, :l+1]))
                
            if scale == 0.0:
                # Skip transformation
                e[i] = a[i, l]
            else:
                # Apply Householder transformation
                for k in range(l+1):
                    a[i, k] /= scale
                    h += a[i, k] * a[i, k]
                    
                f = a[i, l]
                g = -np.sign(f) * np.sqrt(h)
                e[i] = scale * g
                h -= f * g
                a[i, l] = f - g
                f = 0.0
                
                for j in range(l+1):
                    # Store u/h in unused part of a
                    a[j, i] = a[i, j] / h
                    
                    # Calculate p = (a·u)/h
                    g = 0.0
                    for k in range(j+1):
                        g += a[j, k] * a[i, k]
                    if l > j:
                        for k in range(j+1, l+1):
                            g += a[k, j] * a[i, k]
                            
                    e[j] = g / h
                    f += e[j] * a[i, j]
                
                hh = f / (h + h)
                
                # Calculate reduced a
                for j in range(l+1):
                    f = a[i, j]
                    g = e[j] - hh * f
                    e[j] = g
                    
                    for k in range(j+1):
                        a[j, k] -= (f * e[k] + g * a[i, k])
                
            d[i] = h
        
        # Store diagonal elements
        for i in range(1, n):
            d[i] = a[i, i]
        
        # Generate transformation matrix
        for i in range(n-2, -1, -1):
            if d[i+1] != 0.0:
                for j in range(i+1):
                    g = 0.0
                    for k in range(i+1):
                        g += a[i+1, k] * z[k, j]
                    for k in range(i+1):
                        z[k, j] -= g * a[i+1, k]
        
        # Store diagonal elements
        for i in range(n):
            d[i] = a[i, i]
            a[i, i] = 1.0
            if i > 0:
                a[i, i-1] = 0.0
                
        return d, e, z
    
    def diagonalize(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize a symmetric matrix using Householder reduction followed by QL algorithm.
        Combines tred2e and tql2e operations.
        
        Args:
            matrix: Symmetric matrix to diagonalize
            
        Returns:
            Tuple containing:
            - Eigenvalues
            - Eigenvectors
        """
        # First reduce to tridiagonal form
        d, e, z = self.tred2e(matrix)
        
        # Then apply QL algorithm with implicit shifts
        eigenvals, eigenvecs = self.tql2e(d, e, z)
        
        return eigenvals, eigenvecs
    
    def matrix_power(self, matrix: np.ndarray, power: float) -> np.ndarray:
        """
        Calculate matrix power using eigendecomposition
        Equivalent to dmpower Fortran routine
        """
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        return np.dot(eigenvecs * np.power(eigenvals, power), eigenvecs.T)
    
    def matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate matrix inverse
        Equivalent to dmatinv Fortran routine
        """
        return np.linalg.inv(matrix)
    
    def rotate_coordinates(self, coords: np.ndarray, 
                         rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Rotate coordinates using rotation matrix
        Based on rotn Fortran subroutine
        """
        return np.dot(coords, rotation_matrix.T)
    
    def tql2e(self, d: np.ndarray, e: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find eigenvalues and eigenvectors of symmetric tridiagonal matrix using QL method.
        Python implementation of the Fortran tql2e subroutine.
        
        Args:
            d: Diagonal elements from tred2e
            e: Off-diagonal elements from tred2e
            z: Transformation matrix from tred2e
        
        Returns:
            Tuple containing:
            - Eigenvalues
            - Eigenvectors
        """
        n = len(d)
        e = e.copy()  # Make a copy since we'll modify it
        
        machep = np.finfo(np.float64).eps
        
        for l in range(1, n):
            e[l-1] = e[l]
        e[n-1] = 0.0
        
        f = 0.0
        tst1 = 0.0
        
        for l in range(n):
            # Find small subdiagonal element
            tst1 = max(tst1, abs(d[l]) + abs(e[l]))
            m = l
            while m < n:
                if abs(e[m]) <= machep * tst1:
                    break
                m += 1
            
            # If m == l, d[l] is an eigenvalue
            # Otherwise, iterate
            if m > l:
                iter_count = 0
                while True:
                    iter_count += 1
                    if iter_count > 30:
                        raise RuntimeError("No convergence to an eigenvalue after 30 iterations")
                    
                    # Compute implicit shift
                    g = d[l]
                    p = (d[l+1] - g) / (2.0 * e[l])
                    r = np.sqrt(p*p + 1.0)
                    d[l] = e[l] / (p + np.sign(p) * r)
                    h = g - d[l]
                    
                    for i in range(l+1, n):
                        d[i] -= h
                    
                    f += h
                    
                    # QL transformation
                    p = d[m]
                    c = 1.0
                    s = 0.0
                    
                    for i in range(m-1, l-1, -1):
                        g = c * e[i]
                        h = c * p
                        
                        if abs(p) >= abs(e[i]):
                            c = e[i] / p
                            r = np.sqrt(c*c + 1.0)
                            e[i+1] = s * p * r
                            s = c / r
                            c = 1.0 / r
                        else:
                            c = p / e[i]
                            r = np.sqrt(c*c + 1.0)
                            e[i+1] = s * e[i] * r
                            s = 1.0 / r
                            c = c / r
                        
                        p = c * d[i] - s * g
                        d[i+1] = h + s * (c * g + s * d[i])
                        
                        # Accumulate transformation
                        for k in range(n):
                            h = z[k, i+1]
                            z[k, i+1] = s * z[k, i] + c * h
                            z[k, i] = c * z[k, i] - s * h
                    
                    e[l] = s * p
                    d[l] = c * p
                    
                    # Check for convergence
                    if abs(e[l]) <= machep * tst1:
                        break
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = d.argsort()[::-1]
        d = d[idx]
        z = z[:, idx]
        
        return d, z

class BMatrixOrthogonalization:
    """Handles B-matrix orthogonalization procedures"""
    
    def __init__(self):
        self.eps = np.finfo(float).eps
        
    def orthogonalize_b_matrix(self, b_matrix: np.ndarray, 
                              masses: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Orthogonalize B-matrix using mass-weighted Gram-Schmidt process
        
        Args:
            b_matrix: Original B-matrix
            masses: Optional atomic masses for mass-weighting
            
        Returns:
            Tuple containing:
            - Orthogonalized B-matrix
            - Transformation matrix D (Bo = D @ B)
            - Normalization factors
        """
        n_coords, n_cart = b_matrix.shape
        
        # Initialize matrices
        bo_matrix = b_matrix.copy()  # Orthogonalized B-matrix
        d_matrix = np.eye(n_coords)  # Transformation matrix
        
        # Calculate normalization factors
        bnorm = np.zeros(n_coords)
        for i in range(n_coords):
            if masses is not None:
                # Mass-weight the B-matrix rows
                mass_weights = np.repeat(np.sqrt(masses), 3)
                bo_matrix[i] *= mass_weights
            
            bnorm[i] = np.linalg.norm(bo_matrix[i])
            if bnorm[i] > self.eps:
                bo_matrix[i] /= bnorm[i]
        
        # Gram-Schmidt orthogonalization
        for i in range(n_coords):
            # Remove projections of previous vectors
            for j in range(i):
                proj = np.dot(bo_matrix[i], bo_matrix[j])
                bo_matrix[i] -= proj * bo_matrix[j]
                d_matrix[i] -= proj * d_matrix[j]
            
            # Normalize
            norm = np.linalg.norm(bo_matrix[i])
            if norm > self.eps:
                bo_matrix[i] /= norm
                d_matrix[i] /= norm
        
        return bo_matrix, d_matrix, bnorm
    
    def verify_orthogonality(self, bo_matrix: np.ndarray, 
                            threshold: float = 1e-10) -> Tuple[bool, np.ndarray]:
        """
        Verify orthogonality of B-matrix
        
        Args:
            bo_matrix: Orthogonalized B-matrix
            threshold: Orthogonality threshold
            
        Returns:
            Tuple containing:
            - Boolean indicating if matrix is orthogonal
            - Overlap matrix
        """
        overlap = np.dot(bo_matrix, bo_matrix.T)
        
        # Check diagonal elements are ~1 and off-diagonal elements are ~0
        is_orthogonal = True
        n = len(overlap)
        for i in range(n):
            for j in range(n):
                if i == j:
                    if abs(overlap[i,j] - 1.0) > threshold:
                        is_orthogonal = False
                else:
                    if abs(overlap[i,j]) > threshold:
                        is_orthogonal = False
        
        return is_orthogonal, overlap
    
    def mass_weight_b_matrix(self, b_matrix: np.ndarray, 
                            masses: np.ndarray) -> np.ndarray:
        """
        Apply mass-weighting to B-matrix
        
        Args:
            b_matrix: Original B-matrix
            masses: Atomic masses
            
        Returns:
            Mass-weighted B-matrix
        """
        # Repeat masses for x,y,z components
        mass_weights = np.repeat(np.sqrt(masses), 3)
        return b_matrix * mass_weights[np.newaxis, :]
