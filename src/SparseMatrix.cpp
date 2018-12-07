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
                       diagonal.d_values);
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
                       diagonal.d_values,
                       A.ell_width,
                       A.ell_col_ind,
                       A.ell_val,
                       A.inv_diag);
}

void ConvertToELL(SparseMatrix& A)
{
    // TODO on device

    // Allocate arrays
    HIP_CHECK(hipMalloc((void**)&A.ell_col_ind, sizeof(local_int_t) * A.ell_width * A.localNumberOfRows));
    HIP_CHECK(hipMalloc((void**)&A.ell_val, sizeof(double) * A.ell_width * A.localNumberOfRows));
    HIP_CHECK(hipMalloc((void**)&A.halo_row_ind, sizeof(local_int_t) * A.totalToBeSent));
    HIP_CHECK(hipMalloc((void**)&A.halo_col_ind, sizeof(local_int_t) * A.ell_width * A.totalToBeSent));
    HIP_CHECK(hipMalloc((void**)&A.halo_val, sizeof(double) * A.ell_width * A.totalToBeSent));
    HIP_CHECK(hipMalloc((void**)&A.diag_idx, sizeof(local_int_t) * A.localNumberOfRows));
    HIP_CHECK(hipMalloc((void**)&A.inv_diag, sizeof(double) * A.localNumberOfRows));

    std::vector<local_int_t> ell_col_ind(A.ell_width * A.localNumberOfRows);
    std::vector<double> ell_val(A.ell_width * A.localNumberOfRows);
    std::vector<local_int_t> halo_row_ind(A.totalToBeSent);
    std::vector<local_int_t> halo_col_ind(A.ell_width * A.totalToBeSent);
    std::vector<double> halo_val(A.ell_width * A.totalToBeSent);
    std::vector<double> inv_diag(A.localNumberOfRows);

    local_int_t h = 0;
    for(local_int_t i = 0; i < A.localNumberOfRows; ++i)
    {
        local_int_t j = 0;
        local_int_t p = 0;
        local_int_t q = 0;
        bool flag = false;

        for(; j < A.nonzerosInRow[i]; ++j)
        {
            local_int_t col = A.mtxIndL[i][j];
            double val = A.matrixValues[i][j];

            local_int_t idx = p++ * A.localNumberOfRows + i;
            ell_col_ind[idx] = col;
            ell_val[idx] = val;

            if(col >= A.localNumberOfRows)
            {
                idx = q++ * A.totalToBeSent + h;
                halo_row_ind[h] = i;
                halo_col_ind[idx] = col;
                halo_val[idx] = val;
                flag = true;
            }

            if(col == i)
            {
                inv_diag[i] = 1.0 / val;
            }
        }

        for(; p < A.ell_width; ++p)
        {
            local_int_t idx = p * A.localNumberOfRows + i;
            ell_col_ind[idx] = -1;
            ell_val[idx] = 0.0;
        }

        if(flag == true)
        {
            for(; q < A.ell_width; ++q)
            {
                local_int_t idx = q * A.totalToBeSent + h;
                halo_col_ind[idx] = -1;
                halo_val[idx] = 0.0;
            }

            ++h;
        }
    }

    HIP_CHECK(hipMemcpy(A.ell_col_ind, ell_col_ind.data(), sizeof(local_int_t) * A.ell_width * A.localNumberOfRows, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(A.ell_val, ell_val.data(), sizeof(double) * A.ell_width * A.localNumberOfRows, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(A.halo_row_ind, halo_row_ind.data(), sizeof(local_int_t) * A.totalToBeSent, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(A.halo_col_ind, halo_col_ind.data(), sizeof(local_int_t) * A.ell_width * A.totalToBeSent, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(A.halo_val, halo_val.data(), sizeof(double) * A.ell_width * A.totalToBeSent, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(A.inv_diag, inv_diag.data(), sizeof(double) * A.localNumberOfRows, hipMemcpyHostToDevice));
}
