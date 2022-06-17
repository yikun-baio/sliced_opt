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



/**
 * cost_matrx
 * Calculates the cost matrix for two Arrays using the cost_function defined elsewhere.
 *
 * @param X - (n,) Array -> First data set
 * @param Y - (m,) Array -> Second data set
 * @return - (n*m) Array : M_ij = c(X_i, Y_j) where c is defined by cost_function.
 */
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



/**
 * closest_y_M
 *
 * @param M - (n,m) Array
 * @return - indices of smallest values in the array row-wise.
 */
Array closest_y_M(Array & M){
    auto n = (uint32_t) M.shape(0);

    xt::xarray<uint32_t> argmin_Y = xt::zeros<int32_t>({n});
    for (uint32_t i = 0; i < n; ++i){
        auto tmp = xt::argmin(xt::row(M, i));
        argmin_Y(i) = tmp(i);
    }

    return argmin_Y;
}




/**
 * index_adjust
 * Adds an offset (start) to non-negative value of L.
 *
 * @param L - 1d Array(list) -> Non-negative ints or -1. Labelled as Transportation Plan.
 * @param start - Offset to be added. Must be non-negative.
 */
void index_adjust(intArray & L, uint32_t start){
    uint32_t n = L.shape(0);

    for (uint32_t i=0; i < n; ++i){
        if (L(i) >= 0){
            L(i) = L(i) + start;
        }
    }
}



/**
 * startIndex
 * Calculates startIndex.
 * @param L_pre - (n,1) Array.
 * @return - An array with first value as i_start, and second value as j_start.
 */
intArray startIndex(intArray & L_pre){
    uint32_t i_start = L_pre.shape(0);
    int len = (int) i_start;

    intArray ret = {0,0};

    if (i_start == 0){
        return ret;
    }

    for (int i = len-1 ; i > -1; --i ){
        if (L_pre(i) >= 0){
            ret(0) = len;
            ret(1) = (int32_t) L_pre(i) + 1;
            return ret;
        }
    }

    ret(0) = len;
    return ret;
}




/**
 * unassign_y
 *
 * @param L1
 * @return
 */
intArray unassign_y(intArray & L1){
    auto i_last = L1.size() -1;
    auto j_last = L1(i_last);

    for (size_t k = 0; k < j_last + 1; ++ k){
        auto j = (int)(j_last - k);
        auto i = (int)(i_last - k + 1);

        if (j != L1(L1.size() - 1 - k)){
            intArray ret {i, j};
            return ret;
        }
    }

    intArray ret {0 ,-1};
    return ret;
}