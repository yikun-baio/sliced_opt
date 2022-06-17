#include <iostream>
#include "../include/helpers.h"


int main() {
    Array arr {2.0, 5.0, 7.0,
               2.0, 5.0, 7.0};

//    for (auto i = 0; i < 2; ++i){
//        auto x = xt::argmin(xt::row(arr, i));
//        std::cout << x(0) << std::endl;
//    }
    intArray tmp = xt::view(arr, xt::range(0, arr.size()-1, 1), xt::all());
    std::cout << tmp;
    return 0;
}
