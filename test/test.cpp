
#include <iostream>
#include<cstdlib>
#include<Python.h>
//#include<pybind11.h>


//#include<numpy.h>
import_array();
if (PyErr_Occurred()) {
    std::cerr << "Failed to import numpy Python module(s)." << std::endl;
    return NULL; // Or some suitable return value to indicate failure.
}


// int main()
// {
//     std::cout << "Hello World how are you" << std::endl;
// }