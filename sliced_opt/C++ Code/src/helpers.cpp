//
// Created by Rana Muhammad Shahroz Khan on 17/06/2022.
//

#include "../include/helpers.h"


/**
 * cost_function
 * This cost_function is just the different between x and y squared.
 *
 * @param x - double -> Value from first dataset
 * @param y - double -> Value from second dataset
 * @return - Returns the difference of x and y squared.
 */
double cost_function(double & x, double & y){
    double ret = x - y;
    ret = std::pow(ret, 2);

    return ret;
}




Array cost_matrix(Array & X, Array & Y){
    auto n = (uint32_t) X.shape(0);
    auto m = (uint32_t) Y.shape(0);

    Array M = xt::zeros<T>({n, m});
    
    // Iterator in this case does not work.
//    for (auto iterator = M.begin(); iterator != M.end(); ++iterator){
//        *iterator = cost_function(X[])
//    }


    for (uint32_t i = 0; i < n; ++i){
        for (uint32_t j = 0; j < m; ++j){
            M(i,j) = cost_function(X(i), Y(j));
        }
    }

    return M;
}