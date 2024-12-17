# from: /Library/Developer/CommandLineTools/SDKs/MacOSX14.2.sdk/System/
# Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/
# vecLib.framework/Versions/A/Headers/Sparse/Solve.h

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
const att_type = UInt16 # I'm not sure if this is correct.
# attributes could be packed into a single char.
# it may not matter, though, for alignment reasons.
const ATT_TRANSPOSE = att_type(1)
const ATT_LOWER_TRIANGLE = att_type(0)
const ATT_UPPER_TRIANGLE = att_type(2)
const ATT_ORDINARY = att_type(0)
const ATT_TRIANGULAR = att_type(4)
const ATT_UNIT_TRIANGULAR = att_type(8)
const ATT_SYMMETRIC = att_type(12)


# mutable or no?
struct SparseMatrixStructure
    rowCount::Cint
    columnCount::Cint
    columnStarts::Ptr{Clong}
    rowIndices::Ptr{Cint}
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
    order::Ptr{Cvoid}
    ignoreRowsAndColumns::Ptr{Cvoid}
    # not sure how to write types of C-style function pointers in Julia.
    malloc::Ptr{Cvoid}#(::Csize_t)
    free::Ptr{Cvoid}#(::Ptr{Cvoid})
    reportError::Ptr{Cvoid}#(::Cstring) # assuming null-terminated.
end

mutable struct DenseVector{T<:Union{Cdouble, Cfloat}}
    count::Cint
    data::Ptr{T}
end

mutable struct SparseMatrix{T<:Union{Cdouble, Cfloat}}
    structure::SparseMatrixStructure
    data::Ptr{T}
end

mutable struct DenseMatrix{T<:Union{Cdouble, Cfloat}}
    rowCount::Cint
    columnCount::Cint
    columnStride::Cint
    attributes::att_type
    data::Ptr{T}
end

mutable struct SparseOpaqueSymbolicFactorization
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

mutable struct SparseOpaqueFactorization{T<:Union{Cdouble, Cfloat}}
    status::SparseStatus_t
    attributes::att_type
    symbolicFactorization::SparseOpaqueSymbolicFactorization
    userFactorStorage::Bool
    numericFactorization::Ptr{Cvoid}
    solveWorkspaceRequiredStatic::Csize_t
    solveWorkspaceRequiredPerRHS::Csize_t
end

# ignore for now: anything involving Subfactor, Preconditioner, or IterativeMethod

LIBSPARSE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/libSparse.dylib"


# multiply: last argument = product of prior.
function SparseMultiply_smf_dmf_dmf(arg1, arg2, arg3)
    ccall((:_Z14SparseMultiply18SparseMatrix_Float17DenseMatrix_FloatS0_,LIBSPARSE),
            Nothing, (SparseMatrix{Cfloat}, DenseMatrix{Cfloat}, DenseMatrix{Cfloat}), arg1, arg2, arg3)
end

function SparseMultiply_smf_dvf_dvf(arg1, arg2, arg3)
    ccall((:_Z14SparseMultiply18SparseMatrix_Float17DenseVector_FloatS0_,LIBSPARSE),
            Nothing, (SparseMatrix{Cfloat}, DenseVector{Cfloat}, DenseVector{Cfloat}),
            arg1, arg2, arg3)
end


function SparseMultiply_smd_dmd_dmd(arg1, arg2, arg3)
    ccall((:_Z14SparseMultiply19SparseMatrix_Double18DenseMatrix_DoubleS0_,LIBSPARSE),
            Nothing, (SparseMatrix{Cdouble}, DenseMatrix{Cdouble}, DenseMatrix{Cdouble}),
            arg1, arg2, arg3)
end

function SparseMultiply_smf_dvf_dvf(arg1, arg2, arg3)
    ccall((:_Z14SparseMultiply19SparseMatrix_Double18DenseVector_DoubleS0_,LIBSPARSE),
            Nothing, (SparseMatrix{Cdouble}, DenseVector{Cdouble}, DenseVector{Cdouble}),
            arg1, arg2, arg3)
end

# skip 16 Subfactor ones

function SparseMultiply_d_smd_dmd_dmd(arg1, arg2, arg3, arg4)
    ccall((:_Z14SparseMultiplyd19SparseMatrix_Double18DenseMatrix_DoubleS0_,LIBSPARSE),
            Nothing, (Cdouble, SparseMatrix{Cdouble}, DenseMatrix{Cdouble}, DenseMatrix{Cdouble}),
            arg1, arg2, arg3, arg4)
end

function SparseMultiply_d_smd_dvd_dvd(arg1, arg2, arg3, arg4)
    ccall((:_Z14SparseMultiplyd19SparseMatrix_Double18DenseVector_DoubleS0_,LIBSPARSE),
            Nothing, (Cdouble, SparseMatrix{Cdouble}, DenseVector{Cdouble}, DenseVector{Cdouble}),
            arg1, arg2, arg3, arg4)
end

function SparseMultiply_f_smf_dmf_dmf(arg1, arg2, arg3, arg4)
    ccall((:_Z14SparseMultiplyf18SparseMatrix_Float17DenseMatrix_FloatS0_,LIBSPARSE),
            Nothing, (Cfloat, SparseMatrix{Cfloat}, DenseMatrix{Cfloat}, DenseMatrix{Cfloat}),
            arg1, arg2, arg3, arg4)
end

function SparseMultiply_f_smf_dvf_dvf(arg1, arg2, arg3, arg4)
    ccall((:_Z14SparseMultiplyf18SparseMatrix_Float17DenseVector_FloatS0_,LIBSPARSE),
            Nothing, (Cfloat, SparseMatrix{Cfloat}, DenseVector{Cfloat}, DenseVector{Cfloat}),
            arg1, arg2, arg3, arg4)
end


# multiply adds: last argument += product of prior
function SparseMultiplyAdd_smf_dmf_dmf(arg1, arg2, arg3)
    ccall((:_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseMatrix_FloatS0_,LIBSPARSE),
            Nothing, (SparseMatrix{Cfloat}, DenseMatrix{Cfloat}, DenseMatrix{Cfloat}),
            arg1, arg2, arg3)
end

function SparseMultiplyAdd_smf_dvf_dvf(arg1, arg2, arg3)
    ccall((:_Z17SparseMultiplyAdd18SparseMatrix_Float17DenseVector_FloatS0_,LIBSPARSE),
            Nothing, (SparseMatrix{Cfloat}, DenseVector{Cfloat}, DenseVector{Cfloat}),
            arg1, arg2, arg3)
end

function SparseMultiplyAdd_smd_dmd_dmd(arg1, arg2, arg3)
    ccall((:_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseMatrix_DoubleS0_,LIBSPARSE),
            Nothing, (SparseMatrix{Cdouble}, DenseMatrix{Cdouble}, DenseMatrix{Cdouble}),
            arg1, arg2, arg3)
end

function SparseMultiplyAdd_smd_dvd_dvd(arg1, arg2, arg3)
    ccall((:_Z17SparseMultiplyAdd19SparseMatrix_Double18DenseVector_DoubleS0_,LIBSPARSE),
            Nothing, (SparseMatrix{Cdouble}, DenseVector{Cdouble}, DenseVector{Cdouble}),
            arg1, arg2, arg3)
end

function SparseMultiplyAdd_d_smd_dmd_dmd(arg1, arg2, arg3, arg4)
    ccall((:_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseMatrix_DoubleS0_,LIBSPARSE),
    Nothing, (Cdouble, SparseMatrix{Cdouble}, DenseMatrix{Cdouble}, DenseMatrix{Cdouble}),
    arg1, arg2, arg3, arg4)
end

function SparseMultiplyAdd_d_smd_dvd_dvd(arg1, arg2, arg3, arg4)
    ccall((:_Z17SparseMultiplyAddd19SparseMatrix_Double18DenseVector_DoubleS0_,LIBSPARSE),
    Nothing, (Cdouble, SparseMatrix{Cdouble}, DenseVector{Cdouble}, DenseVector{Cdouble}),
    arg1, arg2, arg3, arg4)
end

function SparseMultiplyAdd_f_smf_dmf_dmf(arg1, arg2, arg3, arg4)
    ccall((:_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseMatrix_FloatS0_,LIBSPARSE),
    Nothing, (Cfloat, SparseMatrix{Cfloat}, DenseMatrix{Cfloat}, DenseMatrix{Cfloat}),
    arg1, arg2, arg3, arg4)
end

function SparseMultiplyAdd_f_smf_dvf_dvf(arg1, arg2, arg3, arg4)
    ccall((:_Z17SparseMultiplyAddf18SparseMatrix_Float17DenseVector_FloatS0_,LIBSPARSE),
    Nothing, (Cfloat, SparseMatrix{Cfloat}, DenseVector{Cfloat}, DenseVector{Cfloat}),
    arg1, arg2, arg3, arg4)
end

# tranposes.
function SparseGetTranspose_smf(arg1)
    ccall((:_Z18SparseGetTranspose18SparseMatrix_Float, LIBSPARSE),
        SparseMatrix{Cfloat}, (SparseMatrix{Cfloat},), arg1)
end

function SparseGetTranspose_smd(arg1)
    ccall((:_Z18SparseGetTranspose19SparseMatrix_Double, LIBSPARSE),
        SparseMatrix{Cdouble}, (SparseMatrix{Cdouble},), arg1)
end

# skipped: 2 subfactor transposes.

function SparseGetTranpose_soff(arg1)
    ccall((:_Z18SparseGetTranspose31SparseOpaqueFactorization_Float,LIBSPARSE),
        SparseOpaqueFactorization{Cfloat}, (SparseOpaqueFactorization{Cfloat},), arg1)
end

function SparseGetTranpose_sofd(arg1)
    ccall((:_Z18SparseGetTranspose32SparseOpaqueFactorization_Double,LIBSPARSE),
        SparseOpaqueFactorization{Cdouble}, (SparseOpaqueFactorization{Cdouble},), arg1)
end

function SparseConvertFromCoordinate_d(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKd,LIBSPARSE),
        SparseMatrix{Cdouble}, 
        (Cint, Cint, Clong, UInt8, att_type, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
        arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function SparseConvertFromCoordinate_d_u(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    ccall((:_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKdPvS4_,LIBSPARSE),
        SparseMatrix{Cdouble}, 
        (Cint, Cint, Clong, UInt8, att_type, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cvoid}, Ptr{Cvoid}),
        arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
end

function SparseConvertFromCoordinate_f(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKf,LIBSPARSE),
        SparseMatrix{Cfloat}, 
        (Cint, Cint, Clong, UInt8, att_type, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}),
        arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function SparseConvertFromCoordinate_f_u(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    ccall((:_Z27SparseConvertFromCoordinateiilh18SparseAttributes_tPKiS1_PKfPvS4_,LIBSPARSE),
        SparseMatrix{Cfloat}, 
        (Cint, Cint, Clong, UInt8, att_type, Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat}, Ptr{Cvoid}, Ptr{Cvoid}),
        arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
end