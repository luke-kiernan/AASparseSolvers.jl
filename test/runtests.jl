using Test
using AASparseSolvers
using SparseArrays

# https://developer.apple.com/documentation/accelerate/creating_sparse_matrices?language=objc
@testset "SparseToMatrix" begin
    rI = Cint[0, 1, 3, 0, 1, 2, 3, 1, 2]
    v = Cdouble[2.0, -0.2, 2.5, 1.0, 3.2, -0.1, 1.1, 1.4, 0.5]
    cS = Clong[0,3,7,9]
    expected = Cdouble[
        2.0  1.0 0.0;
        -0.2 3.2 1.4;
        0.0 -0.1 0.5;
        2.5  1.1 0.0;
    ]
    GC.@preserve rI v cS begin
        x = AASparseSolvers.SparseMatrixStructure(4, 3, pointer(cS), pointer(rI),
                                                    AASparseSolvers.ATT_ORDINARY, 1)
        @test AASparseSolvers.SparseToMatrix(AASparseSolvers.SparseMatrix{Cdouble}(x, pointer(v))) == expected
    end

    cS = Clong[0,2,4,5]
    rI = Cint[0,2,0,1,2]
    v = Cdouble[
        1.0, 0.1, 9.2, 0.3, 0.5, 1.3, 0.2, 1.3, 4.5, #Block 0
        0.2, 0.7, 0.9, 1.8, 1.6, 1.7, 0.8, 0.7, 0.9, #Block 1
        1.5, 2.5, 7.2, 0.2, 0.8, 0.8, 0.3, 0.4, 0.2, #Block 2
        0.2, 0.4, 1.8, 1.6, 1.8, 0.6, 0.5, 4.2, 3.3, #Block 3
        0.2, 0.8, 1.2, 0.4, 0.6, 0.8, 1.8, 1.2, 0.9  #Block 4
    ]
    expected = [
        1.0 0.3 0.2 1.5 0.2 0.3 0.0 0.0 0.0;
        0.1 0.5 1.3 2.5 0.8 0.4 0.0 0.0 0.0;
        9.2 1.3 4.5 7.2 0.8 0.2 0.0 0.0 0.0;
        0.0 0.0 0.0 0.2 1.6 0.5 0.0 0.0 0.0;
        0.0 0.0 0.0 0.4 1.8 4.2 0.0 0.0 0.0;
        0.0 0.0 0.0 1.8 0.6 3.3 0.0 0.0 0.0;
        0.2 1.8 0.8 0.0 0.0 0.0 0.2 0.4 1.8;
        0.7 1.6 0.7 0.0 0.0 0.0 0.8 0.6 1.2;
        0.9 1.7 0.9 0.0 0.0 0.0 1.2 0.8 0.9
    ]
    GC.@preserve rI v cS begin
        x = AASparseSolvers.SparseMatrixStructure(3, 3, pointer(cS), pointer(rI),
                                                        AASparseSolvers.ATT_ORDINARY, 3)
        @test AASparseSolvers.SparseToMatrix(AASparseSolvers.SparseMatrix{Cdouble}(x, pointer(v))) == expected
    end

    # test a symmetric one.
    cS = Clong[0, 3, 6, 7, 8]; 
    rI = Cint[0, 1, 3, 1, 2, 3, 2, 3];
    v = Cdouble[10.0, 1.0, 2.5, 12.0, -0.3, 1.1, 9.5, 6.0]
    kind = AASparseSolvers.ATT_SYMMETRIC | AASparseSolvers.ATT_LOWER_TRIANGLE
    expected = [
        10.0  1.0   0.0  2.5;
        1.0   12.0 -0.3  1.1;
        0.0  -0.3   9.5  0.0;
        2.5   1.1   0.0  6.0
    ]
    GC.@preserve rI v cS begin
        x = AASparseSolvers.SparseMatrixStructure(4,4,pointer(cS), pointer(rI), kind, 1)
        @test(AASparseSolvers.SparseToMatrix(AASparseSolvers.SparseMatrix{Cdouble}(x, pointer(v))) == expected)
        v = Cfloat[10.0, 1.0, 2.5, 12.0, -0.3, 1.1, 9.5, 6.0]
        @test(AASparseSolvers.SparseToMatrix(AASparseSolvers.SparseMatrix{Cfloat}(x, pointer(v))) == Cfloat.(expected))
    end
end
@testset "Basic matrix multiply" begin
    dense_data = rand(Float32, 3,3)
    dense2_data = zeros(Float32, 3,3)
    columnStarts = Clong[0, 2, 4, 5]
    rowIndices = Cint[0, 2, 0, 1, 2]
    sparse_data = Cfloat[1.0, 0.1, 9.2, 0.3, 0.5, 1.3, 0.2, 1.3, 4.5]
    
    GC.@preserve dense_data dense2_data columnStarts rowIndices sparse_data begin
        dense = AASparseSolvers.DenseMatrix{Cfloat}(3, 3, 3, 0, pointer(dense_data))
        dense2 = AASparseSolvers.DenseMatrix{Cfloat}(3, 3, 3, 0, pointer(dense2_data))
        s = AASparseSolvers.SparseMatrixStructure(3, 3,
            pointer(columnStarts), pointer(rowIndices),
            AASparseSolvers.ATT_ORDINARY, 1
        )
        sparse_matrix = AASparseSolvers.SparseMatrix{Cfloat}(s, pointer(sparse_data))

        AASparseSolvers.SparseMultiply(sparse_matrix, dense, dense2)
        @test dense2_data ≈ AASparseSolvers.SparseToMatrix(sparse_matrix) * dense_data
    end
end 

#=@testset "Sparse matrix SparseMatrixHacky" begin
    dense_data = rand(Float32, 3,3)
    dense2_data = zeros(Float32, 3,3)
    columnStarts = Clong[0, 2, 4, 5]
    rowIndices = Cint[0, 2, 0, 1, 2]
    sparse_data = Cfloat[1.0, 0.1, 9.2, 0.3, 0.5, 1.3, 0.2, 1.3, 4.5]

    GC.@preserve dense_data dense2_data columnStarts rowIndices sparse_data begin
        dense = AASparseSolvers.DenseMatrix{Cfloat}(3, 3, 3, 0, pointer(dense_data))
        dense2 = AASparseSolvers.DenseMatrix{Cfloat}(3, 3, 3, 0, pointer(dense2_data))
        sparse_matrix = AASparseSolvers.SparseMatrixHacky{Cfloat}(3, 3, pointer(columnStarts), pointer(rowIndices), AASparseSolvers.ATT_ORDINARY, 1, pointer(sparse_data))

        AASparseSolvers.SparseMultiply(sparse_matrix, dense, dense2)
        @test dense2_data ≈ AASparseSolvers.SparseToMatrix(sparse_matrix) * dense_data
    end
end=#

@testset "Cconvert DenseMatrix" begin
    dense = rand(Float32, 3,3)
    dense2 = zeros(Float32, 3, 3)
    columnStarts = Clong[0, 2, 4, 5]
    rowIndices = Cint[0, 2, 0, 1, 2]
    sparse_data = Cfloat[1.0, 0.1, 9.2, 0.3, 0.5, 1.3, 0.2, 1.3, 4.5]
    
    GC.@preserve dense columnStarts rowIndices sparse_data begin
        s = AASparseSolvers.SparseMatrixStructure(3, 3,
                pointer(columnStarts), pointer(rowIndices),
                0, 1)
        sparse_matrix = AASparseSolvers.SparseMatrix{Cfloat}(s, pointer(sparse_data))
        AASparseSolvers.SparseMultiply(sparse_matrix, dense, dense2)
        @test dense2 ≈ AASparseSolvers.SparseToMatrix(sparse_matrix) * dense
    end
end
