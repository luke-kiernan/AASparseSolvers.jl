include("src/wrappers.jl")
dense_data = zeros(Float32, 3,3)
dense = AASparseSolvers.DenseMatrix{Cfloat}(3,3, 3, 0, pointer(dense_data))

dense2_data = ones(Float32, 3,3)
dense2 = AASparseSolvers.DenseMatrix{Cfloat}(3, 3, 3, 0, pointer(dense2_data))

columnStarts = Clong[0, 2, 4, 5]
rowIndices = Cint[0, 2, 0, 1, 2]

matrix_structure = AASparseSolvers.SparseMatrixStructure(
    3, 3, pointer(columnStarts), pointer(rowIndices), 0, 3
)

data = Cfloat[1.0, 0.1, 9.2, 0.3, 0.5, 1.3, 0.2, 1.3, 4.5]
sparse_matrix = AASparseSolvers.SparseMatrix{Cfloat}(matrix_structure, pointer(data))

AASparseSolvers.SparseMultiply(Cfloat(5), sparse_matrix, dense, dense2)
display(dense2_data)