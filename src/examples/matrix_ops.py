# Example usage
matrix_ops = MatrixOperations()

# Create a symmetric matrix
matrix = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 5.0],
    [3.0, 5.0, 6.0]
])

# Get tridiagonal form first
d, e, z = matrix_ops.tred2e(matrix)

# Then find eigenvalues and eigenvectors
eigenvals, eigenvecs = matrix_ops.tql2e(d, e, z)

print("Eigenvalues:", eigenvals)
print("Eigenvectors:", eigenvecs)
