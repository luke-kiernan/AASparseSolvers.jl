using Test
using LinearAlgebra
using AASparseSolvers
using SparseArrays

@testset "SparseMultiply and SparseMultiplyAdd" begin
    # less copy-paste heavy way?
    for T in (Float32, Float64)
        @eval begin
            dense = rand($T, 3,3)
            denseV = rand($T, 3)
            dense2 = zeros($T, 3,3)
            denseV2 = zeros($T, 3)
            sparseM = sprand($T, 3, 3, 0.5)
            sparse_data = sparseM.nzval
            col_inds =  Clong.(sparseM.colptr .+ -1)
            row_inds = Cint.(sparseM.rowval .+ -1)
            GC.@preserve col_inds row_inds sparse_data begin
                s = AASparseSolvers.SparseMatrixStructure(3, 3,
                    pointer(col_inds), pointer(row_inds),
                    AASparseSolvers.ATT_ORDINARY, 1
                )
                sparse_matrix = AASparseSolvers.SparseMatrix{$T}(s, pointer(sparse_data))
                AASparseSolvers.SparseMultiply(sparse_matrix, dense, dense2)
                @test dense2 ≈ sparseM * dense
                AASparseSolvers.SparseMultiply(sparse_matrix, denseV, denseV2)
                @test denseV2 ≈ sparseM * denseV
                scalar = rand($T)
                AASparseSolvers.SparseMultiply(scalar, sparse_matrix, dense, dense2)
                @test dense2 ≈ scalar * sparseM * dense
                AASparseSolvers.SparseMultiply(scalar, sparse_matrix, denseV, denseV2)
                @test denseV2 ≈ scalar * sparseM * denseV
                fill!(dense2, 1.0)
                fill!(denseV2, 1.0)
                AASparseSolvers.SparseMultiplyAdd(sparse_matrix, dense, dense2)
                @test dense2≈ ones(size(dense2))+sparseM * dense
                AASparseSolvers.SparseMultiplyAdd(sparse_matrix, denseV, denseV2)
                @test denseV2 ≈ ones(size(denseV2))+sparseM * denseV
                fill!(dense2, 1.0)
                fill!(denseV2, 1.0)
                scalar = rand($T)
                AASparseSolvers.SparseMultiplyAdd(scalar, sparse_matrix, dense, dense2)
                @test dense2 ≈ ones(size(dense2))+ scalar * sparseM * dense
                AASparseSolvers.SparseMultiplyAdd(scalar, sparse_matrix, denseV, denseV2)
                @test denseV2 ≈ ones(size(denseV2))+ scalar * sparseM * denseV
            end

        end
    end
end

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

@testset "AAFactorization *" begin
    for T in (Float32, Float64)
        @eval begin
            jlA = sprand($T, 3, 3, 0.5)
            x = rand($T, 3, 3)
            aaA = AAFactorization(jlA)
            @test aaA * x ≈ Array(jlA) * x
        end
    end
end

@testset "AAFactorization solve" begin
    for T in (Float32, Float64)
        @eval begin
            N = 3
            jlA = sprand($T, N, N, 0.5)
            while det(jlA) ≈ 0
                global jlA = sprand($T, N, N, 0.5)
            end
            test_fact = AAFactorization(jlA)
            B = rand($T, N, N)
            @test solve(test_fact, B) ≈ Array(jlA) \ B
            b = rand($T, N)
            @test solve(test_fact, b) ≈ Array(jlA) \ b
        end
    end
end

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

@testset "ccall sparsefactor symbolic on singular" begin
    sparseM = sprand(4, 4, 0.1)
    @assert det(sparseM) == 0
    col_inds = Clong.(sparseM.colptr .+ -1)
    row_inds = Cint.(sparseM.rowval .+ -1)
    qrType = AASparseSolvers.SparseFactorizationQR
    GC.@preserve col_inds row_inds begin
        err = nothing
        s = AASparseSolvers.SparseMatrixStructure(4, 4,
            pointer(col_inds), pointer(row_inds),
            AASparseSolvers.ATT_ORDINARY, 1)
        try
            sf = AASparseSolvers.SparseFactor(qrType, s)
        catch err
        end
        @test err isa Exception
        @test sprint(showerror, err) == "Matrix is structurally singular\n"
    end
end

@testset "ccall sparsefactor" begin
    for T in (Float32, Float64)
        @eval begin
            sparseM = sprand($T, 4, 4, 0.5)
            while det(sparseM) == 0
                global sparseM = sprand($T, 4, 4, 0.5)
            end
            sparse_data = copy(sparseM.nzval)
            col_inds = Clong.(sparseM.colptr .+ -1)
            row_inds = Cint.(sparseM.rowval .+ -1)
            expected = rand($T, 4, 4)
            B = Array(sparseM) * expected
            GC.@preserve sparse_data col_inds row_inds begin
                s = AASparseSolvers.SparseMatrixStructure(4, 4,
                    pointer(col_inds), pointer(row_inds),
                    AASparseSolvers.ATT_ORDINARY, 1)
                sparse_matrix = AASparseSolvers.SparseMatrix{$T}(s,
                                        pointer(sparse_data))
                qrType = AASparseSolvers.SparseFactorizationQR
                sf = AASparseSolvers.SparseFactor(qrType, s)
                @test sf.status == AASparseSolvers.SparseStatusOk
                f = AASparseSolvers.SparseFactor(qrType, sparse_matrix)
                AASparseSolvers.SparseSolve(f, B)
                @test B ≈ expected
                AASparseSolvers.SparseCleanup(f)
                AASparseSolvers.SparseCleanup(sf)
            end
        end
    end
end

@testset "non-square in-place solve" begin
    tallMatrix = sprand(6,3,0.5)
    aa_fact = AAFactorization(tallMatrix)
    x, X = rand(3), rand(3, 3)
    b, B = tallMatrix * x, tallMatrix * X
    solve!(aa_fact, b)
    @test isapprox(b, x; 0.001)
    # solve!(aa_fact, B)
    # @test isapprox(B, X; 0.001)
    shortMatrix = sprand(3,4,0.9)
    aa_fact2 = AAFactorization(shortMatrix)
    x, X = rand(4), rand(4,4)
    b, B = shortMatrix * x, shortMatrix * X
    bx, BX = zeros(4), zeros(4,4)
    bx[1:3], BX[1:3, :] = b, B
    solve!(aa_fact2, bx)
    # Seems like julia and apple make different choices
    # when there's multiple solutions.
    # @test isapprox(bx, shortMatrix\b)
    @test isapprox(shortMatrix * bx, b)
    
    # solve!(aa_fact2, BX)
    # @test BX ≈ X
end

@testset "symmetric LDLT" begin
    N = 4
    temp = sprand(N, N, 0.3)
    A = sparse(temp*temp' + diagm(rand(N)))
    A2 = tril(A)
    sym_fact = AAFactorization(A2, true, false)
    @test (sym_fact.aa_matrix.structure.attributes | AASparseSolvers.ATT_SYMMETRIC) != 0
    x, X = rand(N), rand(N, 4)
    b, B = A*x, A*X
    @test solve(sym_fact, b) ≈ x
    @test solve(sym_fact, B) ≈ X
end