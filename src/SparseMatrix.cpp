#include "SparseMatrix.hpp"

#include <hip/hip_runtime.h>

__global__ void kernel_copy_diagonal(local_int_t m,
                                     local_int_t n,
                                     local_int_t ell_width,
                                     const local_int_t* ell_col_ind,
                                     const double* ell_val,
                                     double* diagonal)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(row >= m)
    {
        return;
    }

    for(local_int_t p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = ell_col_ind[idx];

        if(col >= 0 && col < n)
        {
            if(col == row)
            {
                diagonal[row] = ell_val[idx];
                break;
            }
        }
        else
        {
            break;
        }
    }
}

void HIPCopyMatrixDiagonal(const SparseMatrix& A, Vector& diagonal)
{
    hipLaunchKernelGGL((kernel_copy_diagonal),
                       dim3((A.localNumberOfRows - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       A.localNumberOfRows,
                       A.localNumberOfColumns,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       diagonal.hip);
}

__global__ void kernel_replace_diagonal(local_int_t m,
                                        local_int_t n,
                                        const double* diagonal,
                                        local_int_t ell_width,
                                        const local_int_t* ell_col_ind,
                                        double* ell_val,
                                        double* inv_diag)
{
    local_int_t row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(row >= m)
    {
        return;
    }

    double diag = diagonal[row];

    for(local_int_t p = 0; p < ell_width; ++p)
    {
        local_int_t idx = p * m + row;
        local_int_t col = ell_col_ind[idx];

        if(col >= 0 && col < n)
        {
            if(col == row)
            {
                ell_val[idx] = diag;
                break;
            }
        }
        else
        {
            break;
        }
    }

    inv_diag[row] = 1.0 / diag;
}

void HIPReplaceMatrixDiagonal(SparseMatrix& A, const Vector& diagonal)
{
    hipLaunchKernelGGL((kernel_replace_diagonal),
                       dim3((A.localNumberOfRows - 1) / 1024 + 1),
                       dim3(1024),
                       0,
                       0,
                       A.localNumberOfRows,
                       A.localNumberOfColumns,
                       diagonal.hip,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       A.inv_diag);
}
