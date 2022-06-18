//
// Created by Rana Muhammad Shahroz Khan on 17/06/2022.
//

#include "../include/helpers.h"


FourRet opt_sub(Array & M1, intArray & L1, double Lambda){
    int n1 = (int)M1.shape(0);
    int m1 = (int)M1.shape(1);

    // Initial case, empty Y
    if (m1 == 0){
        auto [cost_sub, L_sub, cost_sub_pre, L_sub_pre] = empty_Y_opt(n1, Lambda);
        return {cost_sub, L_sub, cost_sub_pre, L_sub_pre};
    }

    auto [i_act, j_act] = unassign_y(L1);
    L1 = xt::view(L1, xt::range(0, L1.size() - 1, 1), xt::all());

    if (L1.shape(0) == 0){
        auto [cost_sub, L_sub, cost_sub_pre, L_sub_pre] = one_x_opt(M1, i_act, j_act, Lambda);
        return {cost_sub, L_sub, cost_sub_pre, L_sub_pre};
    }

    intArray L1_inact = xt::view(L1, xt::range(0, i_act, 1), xt::all());
    intArray L1x_inact = xt::arange<int32_t>(0, i_act);
    double cost_inact = xt::sum(matrix_take(M1, L1x_inact, L1_inact))[0];

    if (i_act == n1- 1){
        auto [cost_sub, L_sub, cost_sub_pre, L_sub_pre] = one_x_opt(M1, i_act, j_act, Lambda);
        cost_sub = cost_inact + cost_sub;
        L_sub = xt::concatenate(xtuple(L1_inact, L_sub));
        if (L_sub(i_act) == -1){
            cost_sub_pre = cost_sub;
            return {cost_sub, L_sub, cost_sub_pre, L_sub};
        }
    }

    // Find the optimal d1 plan
    intArray L1_act = xt::view(L1, xt::range(i_act, L1.size(), 1), xt::all());
    intArray L1x_act = xt::arange<int32_t>(i_act, n1);

    intArray tmp = xt::view(L1x_act, xt::range(1, L1x_act.size(), 1), xt::all());
    Array cost_L1 = matrix_take(M1, tmp, L1_act);
    intArray tmp2 = xt::view(L1x_act, xt::range(0, L1x_act.size() - 1, 1), xt::all());
    Array cost_L2 = matrix_take(M1, tmp2, L1_act);

    int s = cost_L2.shape(0);

    Array cost_list = xt::concatenate(xtuple(cost_L1, cost_L2));
    Array cost_d1 = xt::zeros<T>({s+1});

    int i = s;
    for (; i > -1; --i){
        Array tmp3 = xt::view(cost_list, xt::range(i, i + s, 1), xt::all());
        cost_d1(i) = xt::sum(tmp3)[0];

        if(i-1 >= 0 && cost_list(i-1) >= Lambda){
            cost_d1 = xt::view(cost_d1, xt::range(i, cost_d1.size(), 1), xt::all());
            break;
        }
    }

    int index_d1_opt = xt::argmin(cost_d1)[0] + i;
    double cost_d1_opt = xt::amin(cost_d1)[0] + Lambda;
    
}