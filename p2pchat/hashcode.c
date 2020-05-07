//
// Created by Nguyễn Đức Quang on 5/6/20.
//
// This hash algorithm is created by Dan Bernstein.
//

#include "hashcode.h"

unsigned long hashcode(const char *str) {
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}