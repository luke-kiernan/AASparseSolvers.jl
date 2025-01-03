using SparseArrays

@enum SparseTriangle_t::UInt8 begin
    SparseLowerTriangle = 0
    SparseUpperTriangle = 1
end

@enum SparseKind_t::UInt32 begin
    SparseOrdinary = 0
    SparseTriangular = 1
    SparseUnitTriangular = 2
    SparseSymmetric = 4
end

@enum SparseFactorization_t::UInt8 begin
    SparseFactorizationCholesky = 0
    SparseFactorizationLDLT = 1
    SparseFactorizationLDLTUnpivoted = 2
    SparseFactorizationLDLTSBK = 3
    SparseFactorizationLDLTTPP = 4
    SparseFactorizationQR = 40
    SparseFactorizationCholeskyAtA = 41
end

@enum SparseOrder_t::UInt8 begin
    SparseOrderDefault = 0
    SparseOrderUser = 1
    SparseOrderAMD = 2
    SparseOrderMetis = 3
    SparseOrderCOLAMD = 4
end

@enum SparseScaling_t::UInt8 begin
    SpraseScalingDefault = 0
    SparseScalingUser = 1
    SparseScalingQuilibriationInf = 2
end

@enum SparseStatus_t::Int32 begin
    SparseStatusOk = 0
    SparseStatusFailed = -1
    SparseMatrixIsSingular = -2
    SparseInternalError = -3
    SparseParameterError = -4
    SparseStatusReleased = -2147483647
end

@enum SparseControl_t::UInt32 begin
    SparseDefaultControl = 0
end

# can't pack bitflags in raw Julia, so can't implement SparseAttributes
# directly. Workaround with flags:
const att_type = Cuchar # I'm not sure if this is correct.
# it may not matter, though, for alignment reasons.
const ATT_TRANSPOSE = att_type(1)
const ATT_LOWER_TRIANGLE = att_type(0)
const ATT_UPPER_TRIANGLE = att_type(2)
const ATT_ORDINARY = att_type(0)
const ATT_TRIANGULAR = att_type(4)
const ATT_UNIT_TRIANGULAR = att_type(8)
const ATT_SYMMETRIC = att_type(12)

const vTypes = Union{Cfloat, Cdouble}

# this must be immutable, otherwise inlining inside SparseMatrix
# gets messed up.
struct SparseMatrixStructure
    rowCount::Cint
    columnCount::Cint
    columnStarts::Ptr{Clong} # Apple uses different types for indices
    rowIndices::Ptr{Cint} # whereas SparseMatrixCSC uses the same
    attributes::att_type
    blockSize::UInt8
end

struct SparseNumericFactorOptions
    control::SparseControl_t
    scalingMethod::SparseScaling_t
    scaling::Ptr{Cvoid}
    pivotTolerance::Float64
    zeroTolerance::Float64
end

struct SparseSymbolicFactorOptions
    control::SparseControl_t
    orderMethod::SparseOrder_t
    order::Ptr{Cvoid} # maybe this should be Ptr{Int}
    ignoreRowsAndColumns::Ptr{Cvoid}
    malloc::Ptr{Cvoid} # arg: Csize_t
    free::Ptr{Cvoid} # arg: Ptr{Cvoid}
    reportError::Ptr{Cvoid} # arg: Cstring, assuming null-terminated.
end

mutable struct DenseVector{T<:vTypes}
    count::Cint
    data::Ptr{T}
end

struct SparseMatrix{T<:vTypes}
    structure::SparseMatrixStructure
    data::Ptr{T}
end

mutable struct DenseMatrix{T<:vTypes}
    rowCount::Cint
    columnCount::Cint
    columnStride::Cint
    attributes::att_type
    data::Ptr{T}
end

# must be immutable.
struct SparseOpaqueSymbolicFactorization
    status::SparseStatus_t 
    rowCount::Cint 
    columnCount::Cint
    attributes::att_type
    blockSize::UInt8
    type::SparseFactorization_t
    factorization::Ptr{Cvoid}
    workspaceSize_Float::Csize_t
    workspaceSize_Double::Csize_t
    factorSize_Float::Csize_t
    factorSize_Double::Csize_t
end
const SymbolicFactor = SparseOpaqueSymbolicFactorization

# TODO: I have T here to match the C: _Double, _Float variants. But T isn't used!!
mutable struct SparseOpaqueFactorization{T<:vTypes}
    status::SparseStatus_t
    attributes::att_type
    symbolicFactorization::SparseOpaqueSymbolicFactorization
    userFactorStorage::Bool
    numericFactorization::Ptr{Cvoid}
    solveWorkspaceRequiredStatic::Csize_t
    solveWorkspaceRequiredPerRHS::Csize_t
end

const Factor{T} = SparseOpaqueFactorization{T}
# ignore for now: anything involving Subfactor, Preconditioner, or IterativeMethod

LIBSPARSE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/libSparse.dylib"

#= function Base.cconvert(::Type{SparseMatrix{T}}, jl::SparseMatrixCSC{T, Int64}) where T<:vTypes
    println("in cconvert")
    sleep(2)
    reindexedCols = jl.colptr .+ -1
    reindexedRows = Cint.(jl.rowval .+ -1)
    return (jl, SparseMatrix{T}(
        SparseMatrixStructure(jl.m, jl.n, C_NULL, C_NULL, ATT_ORDINARY, 1),
        C_NULL
    ), reindexedCols, reindexedRows)
end

function Base.unsafe_convert(::Type{SparseMatrix{T}}, tup::Tuple{SparseMatrixCSC{T}, SparseMatrix{T},
                                                                Vector{Clong}, Vector{Cint}}) where T<:vTypes
    println("in unsafe_convert")
    sleep(2)
    jl, apple, c, r = tup
    # I think it's failing because this is a new allocation, when
    # "all allocations of memory that will be accessed by the C code"
    # should be performed inside cconvert.
    # SparseMatrixStructure must be immutable, though, so there's no
    # easy way to change the fields
    apple.structure = SparseMatrixStructure(jl.m, jl.n,
                                pointer(c), pointer(r),
                                ATT_ORDINARY, 1)
    apple.data = pointer(jl.nzval)
    return apple
end=#

function Base.cconvert(::Type{DenseMatrix{T}}, m::Matrix{T}) where T<:vTypes
    rowCount, colCount = size(m)
    dm = DenseMatrix{T}(rowCount, colCount, rowCount, ATT_ORDINARY, C_NULL)
    return (m, dm)
end

function Base.unsafe_convert(::Type{DenseMatrix{T}}, tup::Tuple{Matrix{T}, DenseMatrix{T}}) where T<:vTypes
    m, dm = tup
    dm.data = pointer(m)
    return dm
end

function Base.cconvert(::Type{DenseVector{T}}, v::Vector{T}) where T<:vTypes
    return (v, DenseVector{T}(size(v)[1], C_NULL))
end

function Base.unsafe_convert(::Type{DenseVector{T}}, tup::Tuple{Vector{T}, DenseVector{T}}) where T<:vTypes
    v, a = tup
    a.data = pointer(v)
    return a
end

for T in (Cfloat, Cdouble)
    # TODO: simpler way that doesn't require @eval?
    # also, if I could write "dense vector-or-matrix," that'd cut the redundancy in half.
    # TODO: find a way to avoid copy-pasting Union{DenseMatrix{$T},Matrix{$T}} over and over.
    # (Maybe I could just simplify the interface and only call with DenseMatrix{$T} now
    # that things are working.)
    local dmMultMangled = T == Cfloat ? :_Z14SparseMultiply18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                        :_Z14SparseMultiply19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::SparseMatrix{$T}, arg2::Union{DenseMatrix{$T},Matrix{$T}},
                                                            arg3::Union{DenseMatrix{$T},Matrix{$T}})
            @ccall LIBSPARSE.$dmMultMangled(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T},
                                                arg3::DenseMatrix{$T})::Cvoid
        end
    end

    #=@eval begin
        function SparseMultiply(arg1::SparseMatrixCSC{$T}, arg2::Matrix{$T}, arg3::Matrix{$T})
            @ccall LIBSPARSE.$dmMultMangled(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T},
                                                arg3::DenseMatrix{$T})::Cvoid
        end
    end=#

    local dvMultMangled = T == Cfloat ? :_Z14SparseMultiply18SparseMatrix_Float17DenseVector_FloatS0_ :
                        :_Z14SparseMultiply19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::SparseMatrix{$T}, arg2::Union{DenseVector{$T},Vector{$T}},
                                                        arg3::Union{DenseVector{$T},Vector{$T}})
            @ccall LIBSPARSE.$dvMultMangled(arg1::SparseMatrix{$T}, arg2::DenseVector{$T},
                                            arg3::DenseVector{$T})::Cvoid
        end
    end

    local sdmMultMangled = T == Cfloat ? :_Z14SparseMultiplyf18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                :_Z14SparseMultiplyd19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::$T, arg2::SparseMatrix{$T}, arg3::Union{DenseMatrix{$T},Matrix{$T}},
                                                    arg4::Union{DenseMatrix{$T},Matrix{$T}})
            @ccall LIBSPARSE.$sdmMultMangled(arg1::$T, arg2::SparseMatrix{$T},
                                                arg3::DenseMatrix{$T}, arg4::DenseMatrix{$T})::Cvoid
        end
    end

    local sdvMultMangled = T == Cfloat ? :_Z14SparseMultiplyf18SparseMatrix_Float17DenseVector_FloatS0_ :
                                        :_Z14SparseMultiplyd19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiply(arg1::$T, arg2::SparseMatrix{$T}, arg3::Union{DenseVector{$T}, Vector{$T}},
                                                    arg4::Union{DenseVector{$T}, Vector{$T}})
            @ccall LIBSPARSE.$sdvMultMangled(arg1::$T, arg2::SparseMatrix{$T},
                                                arg3::DenseVector{$T}, arg4::DenseVector{$T})::Cvoid
        end
    end

    local dmMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                            :_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg1::SparseMatrix{$T}, arg2::Union{DenseMatrix{$T},Matrix{$T}},
                                                arg3::Union{DenseMatrix{$T},Matrix{$T}})
            @ccall LIBSPARSE.$dmMultAddMangled(arg1::SparseMatrix{$T}, arg2::DenseMatrix{$T},
                                                arg3::DenseMatrix{$T})::Cvoid
        end
    end

    local dvMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseVector_FloatS0_ :
                                            :_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg1::SparseMatrix{$T}, arg2::Union{DenseVector{$T},Vector{$T}},
                                                                arg3::Union{DenseVector{$T},Vector{$T}})
            @ccall LIBSPARSE.$dvMultAddMangled(arg1::SparseMatrix{$T}, arg2::DenseVector{$T},
                                                arg3::DenseVector{$T})::Cvoid
        end
    end

    local sdmMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseMatrix_FloatS0_ :
                                        :_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseMatrix_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg0::$T, arg1::SparseMatrix{$T}, arg2::Union{DenseMatrix{$T},Matrix{$T}},
                                                                arg3::Union{DenseMatrix{$T},Matrix{$T}})
            @ccall LIBSPARSE.$sdmMultAddMangled(arg0::$T, arg1::SparseMatrix{$T},
                                                arg2::DenseMatrix{$T}, arg3::DenseMatrix{$T})::Cvoid
        end
    end

    local sdvMultAddMangled = T == Cfloat ? :_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseVector_FloatS0_ :
                                        :_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseVector_DoubleS0_
    @eval begin
        function SparseMultiplyAdd(arg0::$T, arg1::SparseMatrix{$T}, arg2::Union{DenseVector{$T},Vector{$T}},
                                                                        arg3::Union{DenseVector{$T},Vector{$T}})
            @ccall LIBSPARSE.$sdvMultAddMangled(arg0::$T, arg1::SparseMatrix{$T},
                                                arg2::DenseVector{$T}, arg3::DenseVector{$T})::Cvoid
        end
    end

    local mTransposeMangled = T == Cfloat ? :_Z18SparseGetTranspose18SparseMatrix_Float :
                                            :_Z18SparseGetTranspose19SparseMatrix_Double
    @eval begin
        function SparseGetTranspose(arg1::SparseMatrix{$T})
            @ccall LIBSPARSE.$mTransposeMangled(arg1::SparseMatrix{$T})::SparseMatrix{$T}
        end
    end
    # skipped: 2 subfactor transposes.
    local ofTransposeMangled = T == Cfloat ? :_Z18SparseGetTranspose31SparseOpaqueFactorization_Float :
                                        :_Z18SparseGetTranspose32SparseOpaqueFactorization_Double
    @eval begin
        function SparseGetTranspose(arg1::SparseOpaqueFactorization{$T})
            @ccall LIBSPARSE.$ofTransposeMangled(arg1::SparseOpaqueFactorization{$T})::SparseOpaqueFactorization{$T}
        end
    end

    local convertCoordMangled = T == Cfloat ? :_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKf :
                                            :_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKd
    @eval begin
        function SparseConvertFromCoord(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                                        arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T})
            @ccall LIBSPARSE.$convertCoordMangled(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                                                    arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T})::SparseMatrix{$T}
        end
    end

    local uConvertCoordMangled = T == Cfloat ? :_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKdPvS4_ :
                                            :_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKfPvS4_
    @eval begin
        function SparseConvertFromCoord(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                    arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T}, arg9::Ptr{Cvoid}, arg10::Ptr{Cvoid})
            @ccall LIBSPARSE.$uConvertCoordMangled(arg1::Cint, arg2::Cint, arg3::Clong, arg4::Cuchar, arg5::$att_type,
                    arg6::Ptr{Cint}, arg7::Ptr{Cint}, arg8::Ptr{$T}, arg9::Ptr{Cvoid}, arg10::Ptr{Cvoid})::SparseMatrix{$T}
        end
    end

    local mCleanup = T == Cfloat ? :_Z13SparseCleanup18SparseMatrix_Float :
                                     :_Z13SparseCleanup19SparseMatrix_Double
    @eval SparseCleanup(arg1::SparseMatrix{$T}) = @ccall (
                                LIBSPARSE.$mCleanup(arg1::SparseMatrix{$T}))::Cvoid
    
    local ofCleanup = T == Cfloat ? :_Z13SparseCleanup31SparseOpaqueFactorization_Float :
                                    :_Z13SparseCleanup32SparseOpaqueFactorization_Double
    @eval SparseCleanup(arg1::SparseOpaqueFactorization{$T}) = @ccall (
            LIBSPARSE.$ofCleanup(arg1::SparseOpaqueFactorization{$T}))::Cvoid
    
    # TODO: SparseFactor, on SparseMatrix{T} and SparseMatrixStructure.
    # after those, probably do SparseRetain.

    # line 1537 in header.
    local sparseFactorMatrix = T == Cfloat ? :_Z12SparseFactorh18SparseMatrix_Float :
                                                :_Z12SparseFactorh19SparseMatrix_Double
    @eval begin
        function SparseFactor(arg1::SparseFactorization_t, arg2::SparseMatrix{$T})::Factor{$T}
            @ccall LIBSPARSE.$sparseFactorMatrix(arg1::SparseFactorization_t, arg2::SparseMatrix{$T})::SparseOpaqueFactorization{$T}
        end
    end

    # line 1733 in header.
    # local sparseSolve = T == Cfloat ? :
    local sparseSolveInplace = T == Cfloat ? :_Z11SparseSolve31SparseOpaqueFactorization_Float17DenseMatrix_Float :
                                            :_Z11SparseSolve32SparseOpaqueFactorization_Double18DenseMatrix_Double
    @eval SparseSolve(arg1::Factor{$T}, arg2::Union{Matrix{$T}, DenseMatrix{$T}}) = @ccall (
        LIBSPARSE.$sparseSolveInplace(arg1::Factor{$T}, arg2::DenseMatrix{$T})::Cvoid
    )
end
SparseCleanup(arg1::SymbolicFactor) = @ccall (
        LIBSPARSE._Z13SparseCleanup33SparseOpaqueSymbolicFactorization(
                                arg1::SymbolicFactor))::Cvoid
# If you look at SolveImplementation.h, this just calls
# _SparseSymbolicFactorQR(type, &Matrix, &options);
SparseFactor(arg1::SparseFactorization_t, arg2::SparseMatrixStructure) = @ccall (
    LIBSPARSE._Z12SparseFactorh21SparseMatrixStructure(
        arg1::SparseFactorization_t, arg2::SparseMatrixStructure
    )::SymbolicFactor
)
# Maybe this will work better and not produce memory errors??
# But then I need to figure out sensible defaults for the options
SparseFactorQR(arg1::Ref{SparseMatrixStructure}, 
                arg2::Ref{SparseSymbolicFactorOptions}) = @ccall (
    LIBSPARSE._SparseSymbolicFactorQR(SparseFactorizationQR::SparseFactorization_t,
                arg1::Ptr{SparseMatrixStructure},
                arg2::Ptr{SparseSymbolicFactorOptions})::SymbolicFactor
)