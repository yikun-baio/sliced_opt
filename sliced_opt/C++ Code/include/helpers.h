//
// Created by Rana Muhammad Shahroz Khan on 17/06/2022.
//

#ifndef C___CODE_HELPERS_H
#define C___CODE_HELPERS_H

#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmath.hpp>
#include <cmath>

// Some typedefs to make life easier.
typedef double T;
typedef xt::xarray<T> Array;


/**
 * cost_function
 * This cost_function is just the different between x and y squared.
 *
 * @param x - double -> Value from first dataset
 * @param y - double -> Value from second dataset
 * @return - Returns the difference of x and y squared.
 */
double cost_function(double & x, double & y);



Array cost_matrix(Array & X, Array & Y);

#endif //C___CODE_HELPERS_H
