module AASparseSolvers

include("wrappers.jl")

# utility function mostly for testing.
# TODO: convert to a SparseArray instead?
function SparseToMatrix(A::SparseMatrix{T}) where T <: Union{Cdouble, Cfloat}
    s = A.structure
    rows = s.blockSize*s.rowCount
    cols = s.blockSize*s.columnCount
    B = zeros(T, rows, cols)
    blockSize = s.blockSize
    for j in 1:s.columnCount
        # +1 for 0-indexed => 1-indexed.
        firstBlock = unsafe_load(s.columnStarts, j)+1
        lastBlock = unsafe_load(s.columnStarts, j+1)
        for k in firstBlock:lastBlock
            block_row = unsafe_load(s.rowIndices, k)
            # -1 for 1 indexed => 0-indexed.
            offset = blockSize*blockSize*(k-1)
            for (i,l) in Iterators.product(1:blockSize, 1:blockSize)
                offset += 1 # before we use the value: 0-indexed => 1-indexed.
                # j-1 for 1 indexed => 0 indexed.
                B[block_row*blockSize+i, (j-1)*blockSize+l] = unsafe_load(A.data, offset)
            end
        end
    end
    # If marked as symmetric, make symmetric.
    if s.attributes == ATT_SYMMETRIC | ATT_LOWER_TRIANGLE
        for c in 1:cols*blockSize
            for r in 1:c
                @assert(B[r,c] == 0.0 || B[r,c] == B[c,r])
                B[r,c] = B[c,r]
            end
        end
    elseif s.attributes == ATT_SYMMETRIC | ATT_UPPER_TRIANGLE
        for c in 1:cols*blockSize
            for r in c:(rows*blockSize)
                @assert(B[r,c] == 0.0 || B[r,c] == B[c,r])
                B[r,c] = B[c,r]
            end
        end
    end
    return B
end

function DenseToMatrix(A::DenseMatrix{T}) where T <: vTypes
    rows, cols = A.rowCount, A.columnCount
    B = zeros(T, rows, cols)
    r, c = 1, 1
    for i in 1:(rows*cols)
        B[r, c] = unsafe_load(A.data, i)
        if (r == rows)
            c += 1
            r = 1
        else
            r += 1
        end
    end
    return B
end

# do I need this to be mutable?
mutable struct AASparseMatrix{T <: Union{Cfloat, Cdouble}}
    aa_matrix::SparseMatrix{T}
    # these don't *quite* match SparseArrayCSC fields: 1 vs 0 indexing,
    # Cint vs Clong for row indices.
    _col_inds::Vector{Clong}
    _row_inds::Vector{Cint}
    _data::Vector{T}
end

# ignore attributes for now.
function AASparseMatrix{T}(m::Integer, n::Integer, colptr::Union{Vector{Int64}, Vector{Int32}},
                            rowval::Union{Vector{Int64}, Vector{Int32}}, nzval::Vector{T}) where T <: Union{Cfloat, Cdouble}
    c = Clong.(colptr .+ -1)
    r = Cint.(rowval .+ -1)
    s = SparseMatrixStructure(m, n, pointer(c), pointer(r), ATT_ORDINARY, 1)
    m = SparseMatrix{T}(s, pointer(nzval))
    AASparseMatrix{T}(m, c, r, nzval)
end

mutable struct AADenseMatrix{T <: Union{Cfloat, Cdouble}}
    aa_matrix::DenseMatrix{T}
    _data::Vector{T}
end

function AADenseMatrix{T}(m::Integer, n::Integer, data::Vector{T}) where T <: Union{Cfloat, Cdouble}
    AADenseMatrix{T}(DenseMatrix{T}(m, n, n, ATT_ORDINARY, pointer(data)), data)
end

mutable struct AADenseVector{T <: Union{Cfloat, Cdouble}}
    aa_vector::DenseVector{T}
    _data::Vector{T}
end

function AADenseVector{T}(m::Integer, data::Vector{T}) where T<:Union{Cfloat, Cdouble}
    AADenseVector{T}(DenseVector{T}(m, pointer(data)), data)
end

#= function Base.:(*)(A::AASparseMatrix{T}, x::Union{Matrix{T},Vector{T}}) where T<:AASparseSolvers.vTypes
    @assert size(x)[1] == A.aa_matrix.structure.columnCount
    y = Array{T}(undef, A.aa_matrix.structure.rowCount, size(x)[2:end]...)
    if length(size(x)) == 2
        @assert size(y)[2] == size(x)[2]
    end
    SparseMultiply(A.aa_matrix, x, y)
    return y
end=#
export AASparseMatrix, AADenseMatrix, AADenseVector

end # module AASparseSolvers
