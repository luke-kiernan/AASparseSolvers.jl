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

function AASparseMatrix{T}(m::Integer, n::Integer, colptr::Vector{Int64},
                            rowval::Union{Vector{Int64}, Vector{Int32}}, nzval::Vector{T}) where T <: Union{Cfloat, Cdouble}
    c = colptr .+ -1
    r = Cint.(rowval .+ -1)
    s = SparseMatrixStructure(m, n, pointer(c), pointer(r), ATT_ORDINARY, 1)
    m = SparseMatrix{T}(s, pointer(nzval))
    obj = AASparseMatrix{T}(m, c, r, nzval)
    @assert(obj.aa_matrix.data == pointer(obj._data))
    @assert(obj.aa_matrix.structure.columnStarts == pointer(obj._col_inds))
    @assert(obj.aa_matrix.structure.rowIndices == pointer(obj._row_inds))
    return obj
end

# I could also do multiplication by scalar, but there's no point in doing the ccall
# for that one: just multiply the vector.

# keeping this around for troubleshooting reasons.
#=function mult(A::AASparseMatrix{T}, x::S, y::S) where S<:Union{Matrix{T},Vector{T}} where T <:AASparseSolvers.vTypes
    @assert size(x)[1] == A.aa_matrix.structure.columnCount
    @assert size(y)[1] == A.aa_matrix.structure.rowCount
    display(y)
    sleep(2)
    if length(size(x)) == 2
        @assert size(y)[2] == size(x)[2]
    end
    # this C call also goes awry. so allocating y inside the function doesn't seem to be the problem.
    a, b, c = A._col_inds, A._row_inds, A._data
    GC.@preserve a b c SparseMultiply(A.aa_matrix, x, y) # only passing a reference, not the object
    return nothing
end=#
function Base.:(*)(A::AASparseMatrix{T}, x::Union{Matrix{T},Vector{T}}) where T<:AASparseSolvers.vTypes
    @assert size(x)[1] == A.aa_matrix.structure.columnCount
    y = Array{T}(undef, A.aa_matrix.structure.rowCount, size(x)[2:end]...)
    if length(size(x)) == 2
        @assert size(y)[2] == size(x)[2]
    end
    SparseMultiply(A.aa_matrix, x, y)
    return y
end
export AASparseMatrix

end # module AASparseSolvers
