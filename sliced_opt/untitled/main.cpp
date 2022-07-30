#include <string>
#include <iostream>
using namespace std;


string runLengthEncoding(string str) {
    string tmp = "";


    int i = 0;
    while (i < str.size()){
        char x = str[i];
        int counter = 0;

        if (x == str[i]){
            while (x == str[i]){
                ++i;
                counter++;
                if (counter == 9){
                    tmp += to_string(counter) + x;
                    counter = 0;
                }
            }
            if (counter != 0){
                tmp += to_string(counter) + x;
                counter = 0;
            }
        }
        else{
            ++i;
        }
    }


    return tmp;
}

int main(){

    string x = "AAAAAAAAAAAAABBCCCCDD";

    cout << runLengthEncoding(x);
}