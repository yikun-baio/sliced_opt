#include <iostream>
#include "../include/helpers.h"


int main() {
    Array arr {2.0, 5.0, 7.0};

    for (auto i = 0; i < 3; ++i){
        std::cout << arr(i);
    }


    return 0;
}
