
#recursion

'''def fact( n ):
   if n <1:  
       return 1
   else:
       returnNumber = n * fact( n - 1 )
       print(str(n) + '! = ' + str(returnNumber))
       return returnNumber
'''

def F(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return F(n-1)+F(n-2)

F(100)
