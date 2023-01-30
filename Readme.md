
## Install
Required packeges: [numba](https://numba.pydata.org), [pytorch](https://pytorch.org), [PythonOT](https://pythonot.github.io). 

## OPT solver

The file code/sopt/lib_ot.py contains our 1-D optimal partial transport (OPT problem) solver. In particular, it contains the following functions, most of which are accelarated by numba: 
- "solve_opt" (for data in float64) and "solve_opt_32" (for data in flaot32): our 1-D Optimal partial transport (OPT) sovler 
- "opt_lp": linear programming sovler for OPT 
- "sinkhorn_knopp" and "sinkhorn_knopp_32": the Sinkhorn-knopp algorithm for classical OT problem (see [1])
- "sinkhorn_knopp_opt" and "sinkhorn_knopp_opt_32": Sinkhorn-knopp algorithm for OPT problem accelarated by numba 
- "pot" and "pot_32": python implementation of the solver for partial optimal transport problem (algorithm 1 in [2])



## Shape registration experiment 
The file code/sopt/lib_shape.py contains our method for shape registration experiment based on our OPT solver. In particular it contains the following functions: 

- sopt_main and sopt_main_32: our method for shape registration problem based on our 1-D OPT sovler 
- spot_bonneel and spot_bonneel_32: one python implementation of "Fast Iterative Sliced Transport algorithm" based on SPOT in [2]) 
- icp_du and icp_du_32: Du's ICP algorithm (see [3])
- icp_umeyama and icp_umeyama_32: classical ICP algorithm (see [4]) 

Run the file code/experiment/shape_registration/shape_registration_example.ipynb (need to modify the "parant path") to see an example of all methods. 

## color adaptation experiment
The file code/sopt/lib_color.py contains our color adaptation method based on our 1-D OPT solver. In particular, it contains the following functions: 
- sopt_transfer and sopt_transfer_32: our color transporation method based on 1-D OPT solver. 
- spot_transfer and spot_transfer_32: one python implementation of the color transportation method based on SPOT (see [2])
- ot_transfer: one python implementation of OT-based color adaptation method (we modify the code of ot.da.EMDTransport in PythonOT) (see [5])
- eot_transfer: one python implemtation of entropic OT-based color adaptation method (we modify the code of ot.da.SinkhornTransport in PythonOT) (see [5])

Run code/experiment/color_adaptation/color_example.ipynb (need to modify the "parant path") to see an example. 


## References

[1] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013

[2] Nicolas Bonneel and David Coeurjolly. SPOT: sliced partial optimal transport. ACM Transactions on Graphics, 38(4):1–13, 2019

[3] haoyi Du, Nanning Zheng, Shihui Ying, Qubo You, and Yang Wu. An extension of the icp algorithm considering
scale factor. In 2007 IEEE International Conference on Image Processing, volume 5, pages V–193. IEEE, 2007

[4] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions
on Pattern Analysis & Machine Intelligence, 13(04):376–380, 1991.

[5] Ferradans, S., Papadakis, N., Peyre, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.