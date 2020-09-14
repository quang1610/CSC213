# GPU Password Cracker

This project contains 2 CUDA programs, one generates hashcode from an input password and the other uses GPU parallel computing to find the password from input hashcode. A version MD5 hash algorithm is ported to CUDA.

## Installation:

1. Remove the compiled programs if there are any.
```
$ rm password-cracker-gpu
$ rm generate-md5-hash
```  

2. Compile two programs.
```
$ make
```  

3. Generate the hashcode. The input password is limited to 6 charaters.
```
$ ./generate-md5-hash [password]
```  

4. Crack the password. The input hash is a string of Hex with 32 characters. The true MD5 hashcode has 128 bits.
```
$ ./password-cracker-gpu single [hashcode]
```  
The keyword ```single``` indicate that we are cracking a single password.  

## Current bug and future work:
- Implement a feature to crack multiple passwords by reading a list of hashcodes from a file, just like the CPU version.  
- Improve performance of CUDA code. 
- Fix a bug related to endless loops.
