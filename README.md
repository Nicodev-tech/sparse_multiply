void sparse_multiply (
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y  )

Takes two matrixes in Row-Mayor order, x is a 1 x M and A is a N x M matrix.
Then A is transformed into (CSR) form, so that later, 
A in (CSR) form is multiplied by x and the result is in y ( A * x = y )
All of the (CSR) values are stored in user provided buffers in:
out_nnz (amount of non-zero elements), values(non-zero values),
col_indices(index of a non-zero value in the matrix), row_ptrs(beginning of the row in values array)

Transforming the matrix from Row-mayor order to CSR is O(N*M) being N Row length and M Column lenght 
Multiplying A in (CSR) format and x a Column vector takes on worst case O(N^2) if density of A is 50%.  Depending of the definition of sparsed matrixes it could be O(N^2 * M) as A aproaches density of 100%.  Finally if density is < 10% or that the amount of non-zero elements equals to column length or row lenght, is O(N) 
Being N the amount of columns, M the amount of rows and this runs in this O time  assuming that the multiplication is O(1)
To compile the code, run the following command.  
gcc -g -o run challenge.c -lm && ./run