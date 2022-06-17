#include <iostream>
#include "../include/helpers.h"


int main() {
    Array arr {{2.0, 5.0, 7.0},
               {2.0, 5.0, 7.0}};

//    for (auto i = 0; i < 2; ++i){
//        auto x = xt::argmin(xt::row(arr, i));
//        std::cout << x(0) << std::endl;
//    }

    std::cout << arr(-2);
    return 0;
}
