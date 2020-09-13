
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef union uwb {
    unsigned w;
    unsigned char b[4];
} WBunion;

typedef unsigned Digest[4];

typedef unsigned (*DgstFctn)(unsigned a[]);

__device__ void md5(unsigned char *msg, int mlen, unsigned *candidate_hash);