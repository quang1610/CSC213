# CPU Password Cracker

This project cracks password using C and multithreaded programming. Given a hashcode of the password, the algorithm will brute force every possible string to find the string with similar hashcode.  

The hashcode is generated using MD5 algorimth. The length of password is limited to 6 characters.

## Installation:

1. Remove the compiled programs if there are any.
```
$ rm password-cracker
```  

2. Compile two programs.
```
$ make
```  

3. Crack the password. The input hash is a string of Hex with 32 characters. The true MD5 hashcode has 128 bits.
```
$ ./password-cracker single [hashcode]
$ ./password-cracker list [dir to hashcode file]
```  
The keyword ```single``` indicates that we are cracking a single password. The keyword ```list``` indicates that we are cracking multiple passwords by reading their hashcodes from a file. Each line inside the input file has the following format:
```
[username] [hashcode]
```

