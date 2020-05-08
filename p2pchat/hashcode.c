//
// Created by Nguyễn Đức Quang on 5/6/20.
//
// This hash algorithm is created by Dan Bernstein.
//
#include <string.h>
#include "hashcode.h"

unsigned long hashcode(const char *str) {
    unsigned long hash = 5381;
    int len = (int) strlen(str);

    for (int i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + str[i]; /* hash * 33 + c */
    }

    return hash;
}