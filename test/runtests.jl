using Test
using AASparseSolvers
using SparseArrays
using Profile

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
@testset "SparseMultiply and SparseMultiplyAdd" begin
    # less copy-paste heavy way?
    for T in (Float32, Float64)
        @eval begin
            dense_data = rand($T, 3,3)
            denseV_data = rand($T, 3)
            dense2_data = zeros($T, 3,3)
            denseV2_data = zeros($T, 3)
            sparseM = sprand($T, 3, 3, 0.5)
            sparse_data = sparseM.nzval
            col_inds =  Clong.(sparseM.colptr .+ -1)
            row_inds = Cint.(sparseM.rowval .+ -1)
            GC.@preserve dense_data denseV_data dense2_data denseV2_data col_inds row_inds sparse_data begin
                s = AASparseSolvers.SparseMatrixStructure(3, 3,
                    pointer(col_inds), pointer(row_inds),
                    AASparseSolvers.ATT_ORDINARY, 1
                )
                sparse_matrix = AASparseSolvers.SparseMatrix{$T}(s, pointer(sparse_data))
                dense = AASparseSolvers.DenseMatrix{$T}(3, 3, 3, 0, pointer(dense_data))
                dense2 = AASparseSolvers.DenseMatrix{$T}(3, 3, 3, 0, pointer(dense2_data))
                denseV = AASparseSolvers.DenseVector{$T}(3, pointer(denseV_data))
                denseV2 = AASparseSolvers.DenseVector{$T}(3, pointer(denseV2_data))
                AASparseSolvers.SparseMultiply(sparse_matrix, dense, dense2)
                @test dense2_data ≈ sparseM * dense_data
                AASparseSolvers.SparseMultiply(sparse_matrix, denseV, denseV2)
                @test denseV2_data ≈ sparseM * denseV_data
                scalar = rand($T)
                AASparseSolvers.SparseMultiply(scalar, sparse_matrix, dense, dense2)
                @test dense2_data ≈ scalar * sparseM * dense_data
                AASparseSolvers.SparseMultiply(scalar, sparse_matrix, denseV, denseV2)
                @test denseV2_data ≈ scalar * sparseM * denseV_data
            end
            # putting the fills inside the GC.@preserve caused memory errors.
            fill!(dense2_data, 1.0)
            fill!(denseV2_data, 1.0)
            GC.@preserve dense_data denseV_data dense2_data denseV2_data col_inds row_inds sparse_data begin
                AASparseSolvers.SparseMultiplyAdd(sparse_matrix, dense, dense2)
                @test dense2_data ≈ ones(size(dense2_data))+sparseM * dense_data
                AASparseSolvers.SparseMultiplyAdd(sparse_matrix, denseV, denseV2)
                @test denseV2_data ≈ ones(size(denseV2_data))+sparseM * denseV_data
            end

            fill!(dense2_data, 1.0)
            fill!(denseV2_data, 1.0)
            GC.@preserve dense_data denseV_data dense2_data denseV2_data col_inds row_inds sparse_data begin
                scalar = rand($T)
                AASparseSolvers.SparseMultiplyAdd(scalar, sparse_matrix, dense, dense2)
                @test dense2_data ≈ ones(size(dense2_data))+ scalar * sparseM * dense_data
                AASparseSolvers.SparseMultiplyAdd(scalar, sparse_matrix, denseV, denseV2)
                @test denseV2_data ≈ ones(size(denseV2_data))+ scalar * sparseM * denseV_data
            end

        end
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

@testset "cconvert and unsafe_convert dense" begin
    dense = rand(3,3)
    denseV = rand(3)
    dense2 = zeros(3, 3)
    denseV2 = zeros(3)
    sparseM = sprand(3, 3, 0.5)
    sparse_data = sparseM.nzval
    col_inds =  Clong.(sparseM.colptr .+ -1)
    row_inds = Cint.(sparseM.rowval .+ -1)
    # for lack of a way to call cconvert directly, I'll use the matrix multiply routines.
    GC.@preserve dense dense2 denseV denseV2 sparse_data col_inds row_inds begin
        s = AASparseSolvers.SparseMatrixStructure(3, 3,
                pointer(col_inds), pointer(row_inds),
                AASparseSolvers.ATT_ORDINARY, 1)
        sparse_matrix = AASparseSolvers.SparseMatrix{Cdouble}(s, pointer(sparse_data))
        AASparseSolvers.SparseMultiply(sparse_matrix, dense, dense2)
        @test dense2 ≈ sparseM * dense
        AASparseSolvers.SparseMultiply(sparse_matrix, denseV, denseV2)
        @test denseV2 ≈ sparseM * denseV 
    end
end


#=@testset "cconvert and unsafe_convert sparse" begin
    A = SparseMatrixCSC{Float64, Int32}(Array(sprand(Float64, 4, 4, 0.5)))
    x = rand(Float64, 4)
    res = zeros(Float64, 4)
    AASparseSolvers.SparseMultiply(A, x, res)
    @test res ≈ A*x
end=#

# this gives memory errors (sometimes)
#= @testset "AASparseMatrix constructor and *" begin
    for T in (Float32, Float64)
        @eval begin
            jlA = sprand($T, 3, 3, 0.5)
            x = rand($T, 3)
            p1 = jlA.colptr
            p2 = jlA.rowval
            p3 = jlA.nzval
            aaA = AASparseMatrix{$T}(jlA.m, jlA.n, p1, p2, p3)
            GC.@preserve x p1 p2 p3 jlA aaA begin
                @test aaA * x ≈ Array(jlA) * x
            end
        end
    end
end=#

# transpose doesn't actually change the matrix:
# just creates new matrix whose structure has the opposite transpose flag.
@testset "ccall transpose" begin 
    for T in (Float32, Float64)
        @eval begin
            sparseM = sprand($T, 3, 4, 0.5)
            sparse_data = sparseM.nzval
            col_inds =  Clong.(sparseM.colptr .+ -1)
            row_inds = Cint.(sparseM.rowval .+ -1)
            GC.@preserve sparse_data col_inds row_inds begin
                s = AASparseSolvers.SparseMatrixStructure(3, 4,
                        pointer(col_inds), pointer(row_inds),
                        AASparseSolvers.ATT_ORDINARY, 1)
                sparse_matrix = AASparseSolvers.SparseMatrix{$T}(s,
                                        pointer(sparse_data))
                ATT_TRANSP = AASparseSolvers.ATT_TRANSPOSE
                res = AASparseSolvers.SparseGetTranspose(sparse_matrix)
                @test res.data == sparse_matrix.data &&
                            (res.structure.attributes & ATT_TRANSP != zero(typeof(ATT_TRANSP)))
                original = AASparseSolvers.SparseGetTranspose(res)
                @test original.data == sparse_matrix.data &&
                        (original.structure.attributes & ATT_TRANSP == zero(typeof(ATT_TRANSP)))
            end
        end
    end
end

# This gives memory errors (usually)

@testset "ccall sparsefactor " begin
    for T in (Float32, Float64)
        @eval begin
            sparseM = sprand($T, 4, 4, 0.5)
            sparse_data = sparseM.nzval
            col_inds = Clong.(sparseM.colptr .+ -1)
            row_inds = Cint.(sparseM.rowval .+ -1)
            expected = rand($T, 4, 4)
            B = Array(sparseM) * expected
            GC.@preserve sparse_data col_inds row_inds B begin
                s = AASparseSolvers.SparseMatrixStructure(4, 4,
                    pointer(col_inds), pointer(row_inds),
                    AASparseSolvers.ATT_ORDINARY, 1)
                sparse_matrix = AASparseSolvers.SparseMatrix{$T}(s,
                                        pointer(sparse_data))
                qrType = AASparseSolvers.SparseFactorizationQR
                GC.@preserve s sparse_matrix begin
                    f = AASparseSolvers.SparseFactor(qrType, sparse_matrix)
                    GC.@preserve f begin
                        AASparseSolvers.SparseSolve(f, B)
                        @test B ≈ expected
                    end
                end
                # not sure if the memory is Julia-managed or C-managed.
                # AASparseSolvers.SparseCleanup(sf)
            end
        end
    end
end