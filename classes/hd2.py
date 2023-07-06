import numpy as np

class hyperdual2:
    order = 2
    def __init__(self,a:float,b:float=0,c:float=0,d:float=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __repr__(self): # Print statement
        return f'{self.a} + {self.b} \u03B5_1 + {self.c} \u03B5_2 + {self.d} \u03B5_1 \u03B5_2'
    # Basic operations
    def __add__(self,obj2): # Adding 2 hyperdual numbers
        rl = self.a + obj2.a
        i1 = self.b + obj2.b
        i2 = self.c + obj2.c
        i3 = self.d + obj2.d
        return hyperdual2(rl,i1,i2,i3)
    def __sub__(self,obj2): # Subtracting 2 hyperdual numbers
        rl = self.a - obj2.a
        i1 = self.b - obj2.b
        i2 = self.c - obj2.c
        i3 = self.d - obj2.d
        return hyperdual2(rl,i1,i2,i3)
    def __mul__(self,obj2): # Multiply 2 hyperdual numbers
        rl = self.a*obj2.a
        i1 = self.a*obj2.b + self.b*obj2.a
        i2 = self.a*obj2.c + self.c*obj2.a
        i3 = self.a*obj2.d + self.b*obj2.c + self.c*obj2.b + self.d*obj2.a
        return hyperdual2(rl,i1,i2,i3)
    def __truediv__(self,obj2): # Divide 2 hyperdual numbers
        return self*obj2.inv()
    # Comparison
    def __ge__(self,obj2): # Greater than or equals to
        if self.a>=obj2.a:
            return True
        else:
            return False
    def __le__(self,obj2): # Less than or equals to
        if self.a<=obj2.a:
            return True
        else:
            return False
    def __gt__(self,obj2): # Greater than
        if self.a>obj2.a:
            return True
        else:
            return False
    def __lt__(self,obj2): # Less than 
        if self.a<obj2.a:
            return True
        else:
            return False
    def __eq__(self,obj2): # Are 2 entities equal
        if self.a==obj2.a:
            return True
        else:
            return False
    # Active methods    
    def inv(self): # Inverse
        if self.a == 0:
            return f'Not invertable'
        rl = 1/self.a
        i1 = self.b/self.a**2
        i2 = self.c/self.a**2
        i3 = -self.d/self.a**2 + 2*self.b*self.c/self.a**3
        return hyperdual2(rl,i1,i2,i3)