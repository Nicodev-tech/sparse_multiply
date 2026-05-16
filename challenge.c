#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =========================================================
// FUNCTION PROTOTYPE
// =========================================================
void sparse__multiply(
    int rows,
    int cols,
    const double* A,
    const double* x,
    int* out_nnz,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* y
);

// =========================================================
//void sparse_multiply (
//    int rows, int cols, const double* A, const double* x,
//    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
//    double* y  )
//
// Takes two matrixes in Row-Mayor order, x is a 1 x M and A is a N x M matrix.
// Then A is transformed into (CSR) form, so that later, 
// A in (CSR) form is multiplied by x and the result is in y ( A * x = y )
// All of the (CSR) values are stored in user provided buffers in:
// out_nnz (amount of non-zero elements), values(non-zero values),
// col_indices(index of a non-zero value in the matrix), row_ptrs(beginning of the row in values array)
//
// Transforming the matrix from Row-mayor order to CSR
// is O(N*M) being N Row length and M Column lenght
//
// Multiplying A in (CSR) format and x a Column vector takes on average O(
// Being N the Row length
//
// =========================================================
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    int non_zero_elem = 0;
    int start_row = 0;
    int idx_zero_elem = 0;
    //do the 2 steps in one check the non-zero elements and pass them to the CSR format
    for (int i = 0; i < rows; i++)
    {
        //doing this so that in every iteration we don't have to calculate the fixed offset
        int offset = i * cols;
        
        for (int j = 0; j < cols; j++)
        {
            if(A[j + offset] != 0)
            {
                values[non_zero_elem] = A[j + offset];
                col_indices[non_zero_elem] = j;
                non_zero_elem++;

            }
        }

        row_ptrs[idx_zero_elem] = start_row;
        idx_zero_elem += 1;
        start_row = non_zero_elem;
    }
    row_ptrs[idx_zero_elem] = non_zero_elem;
    //could use (out_nnz) directly, but this is clearly more understandable
    (*out_nnz) = non_zero_elem;
    for (int c = 0; c < rows; c++)
    {
        double f_sum = 0.0;
        //Reading only the rows in x with the cols have a non-zero elements in A
        for (int i = row_ptrs[c]; i < row_ptrs[c+1]; i++)
        {
            
            f_sum += x[col_indices[i]] * values[i];
        }
        y[c] = f_sum;
    }
    
}

// =========================================================
// TEST HARNESS
// =========================================================
int main(void) {
    srand(time(NULL));
    
    const int num_iterations = 100;
    int passed_count = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        int rows = rand() % 41 + 5;
        int cols = rand() % 41 + 5;
        double density = 0.05 + (rand() / (double) RAND_MAX) * 0.35;
        
        size_t mat_sz = (size_t) rows * cols;

        double* A = calloc(mat_sz, sizeof(double));
        for (size_t i = 0; i < mat_sz; ++i) {
            if (((double) rand() / RAND_MAX) < density) {
                A[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
            }
        }

        double* values = malloc(mat_sz * sizeof(double));
        int* col_indices = malloc(mat_sz * sizeof(int));
        int* row_ptrs = malloc((rows + 1) * sizeof(int));
        double* x = malloc(cols * sizeof(double));
        double* y_user = malloc(rows * sizeof(double));
        double* y_ref = calloc(rows, sizeof(double));
        int out_nnz = 0;

        for (int i = 0; i < cols; ++i) {
            x[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
        }

        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j) {
                sum += A[i * cols + j] * x[j];
            }
            y_ref[i] = sum;
        }

        sparse_multiply(rows, cols, A, x, 
            &out_nnz, 
            values, 
            col_indices, 
            row_ptrs, y_user);

        double max_err = 0.0;
        int passed = 1;
        for (int i = 0; i < rows; ++i) {
            double diff = fabs(y_user[i] - y_ref[i]);
            double tol = 1e-7 + 1e-7 * fabs(y_ref[i]); // Mixed absolute/relative tolerance
            if (diff > tol) {
                max_err = fmax(max_err, diff);
                passed = 0;
            }
        }

        if (passed) {
            passed_count++;
        }

        printf(
            "Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: %s (Max error: %.2e)\n",
            iter, rows, cols, density, out_nnz, passed ? "PASS" : "FAIL", max_err
        );

        free(A);
        free(values);
        free(col_indices);
        free(row_ptrs);
        free(x);
        free(y_user);
        free(y_ref);
    }

    printf(
        "\n%s (%d/%d iterations passed)\n",
        passed_count == num_iterations ? "All tests passed!" : "Some tests failed.",
        passed_count, num_iterations
    );
           
    return passed_count == num_iterations ? 0 : 1;
}
