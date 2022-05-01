import math
class Solution:
    def intToRoman(self, num: int) -> str:
        self.List1=['I','X','C','M']
        self.List2=['V','L','D','']
        n=len(str(num))-1
        print(n)
        #n=len(self.List1)
        Romannumber=''
        for i in range(n,-1,-1):
            value=10**(i)
            number=num//value
            print(num)
            print('number is '+str(number))
            print('i is '+str(i))
            Roman=self.Roman(number,i)

            Romannumber+=Roman
            print(Roman)
            print(Romannumber)
            num=num%value
        return Romannumber


    def Roman(self,number,i: int)-> str:
        letter1=self.List1[i]
        letter2=self.List2[i]
        if i<=2 and number==4:
            Roman=letter1+letter2
        elif i<=2 and number==9:
            letter0=self.List1[i+1]
            Roman=letter1+letter0
        elif number>=5:
            number=number-5
            Roman=''
            for k in range(number):
                Roman+=letter1
            Roman=letter2+Roman
        elif number<4:
            Roman=''
            for k in range(number):
                Roman+=letter1
                
        return Roman
                
            
                
x=1000
A=Solution().intToRoman(x)
                

                    
