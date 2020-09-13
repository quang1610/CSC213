#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define MD5_UNSIGNED_HASH_LEN 16


// C implementation of MD5 from https://rosettacode.org/wiki/MD5#C
typedef union uwb {
    unsigned w;
    unsigned char b[4];
} WBunion;

typedef unsigned Digest[4];

typedef unsigned (*DgstFctn)(unsigned a[]);

__device__ unsigned f0( unsigned abcd[] ){
    return ( abcd[1] & abcd[2]) | (~abcd[1] & abcd[3]);}

__device__ unsigned f1( unsigned abcd[] ){
    return ( abcd[3] & abcd[1]) | (~abcd[3] & abcd[2]);}

__device__ unsigned f2( unsigned abcd[] ){
    return  abcd[1] ^ abcd[2] ^ abcd[3];}

__device__ unsigned f3( unsigned abcd[] ){
    return abcd[2] ^ (abcd[1] |~ abcd[3]);} 

__device__ unsigned *calcKs( unsigned *k)
{
    double s, pwr;
    int i;

    pwr = pow(2, 32);
    for (i=0; i<64; i++) {
        double temp = 1 + i;
        s = abs(sin(temp));
        k[i] = (unsigned)( s * pwr );
    }
    return k;
}

// ROtate v Left by amt bits
__device__ unsigned rol( unsigned v, short amt )
{
    unsigned  msk1 = (1<<amt) -1;
    return ((v>>(32-amt)) & msk1) | ((v<<amt) & ~msk1);
}

__device__ void md5(unsigned char *msg, int mlen, unsigned char *hash_code)
{
    static Digest h0 = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476 };
//    static Digest h0 = { 0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210 };
    static DgstFctn ff[] = { &f0, &f1, &f2, &f3 };
    static short M[] = { 1, 5, 3, 7 };
    static short O[] = { 0, 1, 5, 0 };
    static short rot0[] = { 7,12,17,22};
    static short rot1[] = { 5, 9,14,20};
    static short rot2[] = { 4,11,16,23};
    static short rot3[] = { 6,10,15,21};
    static short *rots[] = {rot0, rot1, rot2, rot3 };
    static unsigned kspace[64];
    static unsigned *k;

    static Digest h;
    Digest abcd;
    DgstFctn fctn;
    short m, o, g;
    unsigned f;
    short *rotn;
    union {
        unsigned w[16];
        char     b[64];
    }mm;
    int os = 0;
    int grp, grps, q, p;
    unsigned char *msg2;

    if (k==NULL) k= calcKs(kspace);

    for (q=0; q<4; q++) h[q] = h0[q];   // initialize

    {
        grps  = 1 + (mlen+8)/64;
        msg2 = (unsigned char *) malloc( 64*grps);
        memcpy( msg2, msg, mlen);
        msg2[mlen] = (unsigned char)0x80;
        q = mlen + 1;
        while (q < 64*grps){ msg2[q] = 0; q++ ; }
        {
//            unsigned char t;
            WBunion u;
            u.w = 8*mlen;
//            t = u.b[0]; u.b[0] = u.b[3]; u.b[3] = t;
//            t = u.b[1]; u.b[1] = u.b[2]; u.b[2] = t;
            q -= 8;
            memcpy(msg2+q, &u.w, 4 );
        }
    }

    for (grp=0; grp<grps; grp++)
    {
        memcpy( mm.b, msg2+os, 64);
        for(q=0;q<4;q++) abcd[q] = h[q];
        for (p = 0; p<4; p++) {
            fctn = ff[p];
            rotn = rots[p];
            m = M[p]; o= O[p];
            for (q=0; q<16; q++) {
                g = (m*q + o) % 16;
                f = abcd[1] + rol( abcd[0]+ fctn(abcd) + k[q+16*p] + mm.w[g], rotn[q%4]);

                abcd[0] = abcd[3];
                abcd[3] = abcd[2];
                abcd[2] = abcd[1];
                abcd[1] = f;
            }
        }
        for (p=0; p<4; p++)
            h[p] += abcd[p];
        os += 64;
    }

    if( msg2 )
        free( msg2 );

    WBunion u;
    int offset, byte;
    int hash_code_offset = 0;
    for (offset=0; offset<4; offset++){
        u.w = h[offset];
        memcpy(hash_code[hash_code_offset], u.b, sizeof(unsigned char) * 4);
        hash_code_offset += 4;
    }
    hash_code[MD5_UNSIGNED_HASH_LEN] = '\0';
}