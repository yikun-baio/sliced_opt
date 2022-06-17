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

    
}