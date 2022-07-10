# CUDA-Programming
## GPU programming with CUDA as part of CS6023 course in IITM.  
Complete details about the problem statement for each assignment can be found in the respective directories

**Assgt-1**: To compute C = (A + B<sup>T</sup>) * (B<sup>T</sup> - A), where A, B and C are matrices, using per-row-col kernel, per-col-row kernel and per-element kernel

**Assgt-2**: To compute X = (A + B<sup>T</sup>) * C * D<sup>T</sup>, where A, B, C, D, X are matrices, in an optimised way  
*Here I made kernels to compute A * B<sup>T</sup> and A<sup>T</sup> * B in most efficient way. The size of matrices are constrained such that they fit in the limited GPU memory. More details of my solution can be found in attempts/EE19B121.cu*

**Assgt-3**: Multicore Task scheduling problem with compute parallelism  
*Here I used prefix computation to compute the starting time for N different tasks. Note that this and any other method will only result in higher execution time than a simple CPU computation*

**Assgt-4**: Train ticket booking application  
*Here its better to generate as many threads as the number of requests rather than threads for all combination of trains and classes*
