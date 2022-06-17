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
IntRet startIndex(intArray & L_pre){
    uint32_t i_start = L_pre.shape(0);
    int len = (int) i_start;


    if (i_start == 0){
        return {0,0};
    }

    for (int i = len-1 ; i > -1; --i ){
        if (L_pre(i) >= 0){
            return {len, L_pre(i)+1};
        }
    }

    return {len, 0};
}




/**
 * unassign_y
 *
 * @param L1
 * @return
 */
IntRet unassign_y(intArray & L1){
    auto i_last = L1.size() -1;
    auto j_last = L1(i_last);

    for (size_t k = 0; k < (size_t)j_last + 1; ++ k){
        auto j = (int)(j_last - k);
        auto i = (int)(i_last - k + 1);

        if (j != L1(L1.size() - 1 - k)){
            return {i, j};
        }
    }

    return {0, -1};
}




/**
 * empty_Y_opt
 * Generates the plan.
 *
 * @param n - size of the L
 * @return - L -> Optimal plan array.
 */
FourRet empty_Y_opt(int & n, double & Lambda){
    intArray L = xt::zeros<int32_t>({n});

    for (auto iterator = L.begin(); iterator != L.end(); ++iterator){
        *iterator = -1;
    }

    double cost = Lambda * n;

    return {cost, L, cost, L};
}


/**
 * matrix_take
 *
 * @param X
 * @param L1
 * @param L2
 * @return
 */
Array matrix_take(Array & X, Array & L1, Array & L2){
    auto size = L1.shape(0);

    Array ret = xt::zeros<double>({size});

    for (int i = 0; i < (int)size; ++i){
        ret(i) = X(L1(i), L2(i));
    }

    return ret;
}


/**
 * one_x_opt_1
 * Takes care for the return of the first parameter from python version.
 * @param M1
 * @param i_act
 * @param j_act
 * @param Lambda
 * @return
 */
FourRet one_x_opt(Array & M1, int32_t & i_act, int32_t & j_act, double & Lambda){
    if (j_act < 0 ){
        intArray ret = {-1};
        intArray ret1 = {-1};
        return {Lambda, ret, Lambda, ret1};
    }
    auto c_xy = M1(i_act, j_act);
    if (c_xy >= Lambda){
        intArray ret = {-1};
        intArray ret1 = {-1};
        return {Lambda, ret, Lambda, ret1};
    }
    else{
        intArray ret = {j_act};
        intArray ret1 = xt::empty<int32_t>({0});
        return {c_xy, ret, 0, ret1};
    }
}

