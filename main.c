#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

int round_to_multiplier_16(int i) {
    int return_value = 16;

    while (i > return_value) {
        return_value *= 2;
    }

    return return_value;
}

int find_free_list_ind_base16(int i) {
    int return_value = 0;
    while (i > 16) {
        i = i / 2;
        return_value ++;
    }
    return return_value;
}

int main() {
}

