#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
//    for (int i = 0; i <= 2048; i += 100) {
////        printf("malloc %d get size %d\n", i, round_to_multiplier_16(i));
////        printf("round down log %d of 16: %d\n", i, find_free_list_ind_base16(i));
////    }
    for (int i = 0; i < 10; i++) {
        printf("Hi\n");
        while (i < 5) {
            break;
        }
    }

    return 0;
}