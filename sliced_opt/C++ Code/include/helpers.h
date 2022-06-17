//
// Created by Rana Muhammad Shahroz Khan on 17/06/2022.
//

#ifndef C___CODE_HELPERS_H
#define C___CODE_HELPERS_H

#include <tuple>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmath.hpp>
#include <cmath>
#include <xtensor/xsort.hpp>


// Some typedefs to make life easier.
typedef double T;
typedef xt::xarray<T> Array;
typedef xt::xarray<int32_t> intArray;
typedef std::tuple<int32_t, int32_t> IntRet;
typedef std::tuple<T, intArray> OptRet;
typedef std::tuple<T, intArray, T, intArray> FourRet;

/**
 * cost_function
 * This cost_function is just the different between x and y squared.
 *
 * @param x - double -> Value from first dataset
 * @param y - double -> Value from second dataset
 * @return - Returns the difference of x and y squared.
 */
double cost_function(double & x, double & y);


/**
 * cost_matrx
 * Calculates the cost matrix for two Arrays using the cost_function defined elsewhere.
 *
 * @param X - (n,) Array -> First data set
 * @param Y - (m,) Array -> Second data set
 * @return - (n*m) Array : M_ij = c(X_i, Y_j) where c is defined by cost_function.
 */
Array cost_matrix(Array & X, Array & Y);


/**
 * closest_y_M
 *
 * @param M - (n,m) Array
 * @return - indices of smallest values in the array row-wise.
 */
Array closest_y_M(Array & M);



/**
 * index_adjust
 * Adds an offset (start) to non-negative value of L.
 *
 * @param L - 1d Array(list) -> Non-negative ints or -1. Labelled as Transportation Plan.
 * @param start - Offset to be added. Must be non-negative.
 */
void index_adjust(Array & L, uint32_t start = 0);



/**
 * startIndex
 * Calculates startIndex.
 * @param L_pre - (n,1) Array.
 * @return - An array with first value as i_start, and second value as j_start.
 */
IntRet startIndex(Array & L_pre);



/**
 * unassign_y
 *
 * @param L1
 * @return
 */
IntRet unassign_y(intArray & L1);



/**
 * empty_Y_opt
 * Generates the plan.
 *
 * @param n - size of the L
 * @return - L -> Optimal plan array.
 */
FourRet empty_Y_opt(int & n, double  Lambda);


/**
 * matrix_take
 *
 * @param X
 * @param L1
 * @param L2
 * @return
 */
Array matrix_take(Array & X, intArray & L1, intArray & L2);


/**
 * one_x_opt
 *
 * @param M1
 * @param i_act
 * @param j_act
 * @param Lambda
 * @return
 */
FourRet one_x_opt(Array & M1, int32_t & i_act, int32_t & j_act, double & Lambda);



#endif //C___CODE_HELPERS_H
