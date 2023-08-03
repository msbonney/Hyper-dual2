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
        if not isinstance(obj2,hyperdual2):
            obj2 = hyperdual2(obj2)
        rl = self.a + obj2.a
        i1 = self.b + obj2.b
        i2 = self.c + obj2.c
        i3 = self.d + obj2.d
        return hyperdual2(rl,i1,i2,i3)
    def __sub__(self,obj2): # Subtracting 2 hyperdual numbers
        if not isinstance(obj2,hyperdual2):
            obj2 = hyperdual2(obj2)
        rl = self.a - obj2.a
        i1 = self.b - obj2.b
        i2 = self.c - obj2.c
        i3 = self.d - obj2.d
        return hyperdual2(rl,i1,i2,i3)
    def __mul__(self,obj2): # Multiply 2 hyperdual numbers
        if not isinstance(obj2,hyperdual2):
            obj2 = hyperdual2(obj2)
        rl = self.a*obj2.a
        i1 = self.a*obj2.b + self.b*obj2.a
        i2 = self.a*obj2.c + self.c*obj2.a
        i3 = self.a*obj2.d + self.b*obj2.c + self.c*obj2.b + self.d*obj2.a
        return hyperdual2(rl,i1,i2,i3)
    def __truediv__(self,obj2): # Divide 2 hyperdual numbers
        if not isinstance(obj2,hyperdual2):
            obj2 = hyperdual2(obj2)
        return self*obj2.inv()
    # Comparison
    def __ge__(self,obj2): # Greater than or equals to
        if not isinstance(obj2,hyperdual2):
            obj2 = hyperdual2(obj2)
        if self.a>=obj2.a:
            return True
        else:
            return False
    def __le__(self,obj2): # Less than or equals to
        if not isinstance(obj2,hyperdual2):
            obj2 = hyperdual2(obj2)
        if self.a<=obj2.a:
            return True
        else:
            return False
    def __gt__(self,obj2): # Greater than
        if not isinstance(obj2,hyperdual2):
            obj2 = hyperdual2(obj2)
        if self.a>obj2.a:
            return True
        else:
            return False
    def __lt__(self,obj2): # Less than 
        if not isinstance(obj2,hyperdual2):
            obj2 = hyperdual2(obj2)
        if self.a<obj2.a:
            return True
        else:
            return False
    def __eq__(self,obj2): # Are 2 entities equal
        if not isinstance(obj2,hyperdual2):
            obj2 = hyperdual2(obj2)
        if self.a==obj2.a:
            return True
        else:
            return False
    # Active methods    
    def inv(self): # Inverse
        try:
            rl = np.linalg.inv(self.a)
            i1 = -self.b*(rl**2)
            i2 = -self.c*(rl**2)
            i3 = self.d*(rl**2) - 2*self.b*self.c*(rl**3)
        except :
            if self.a == 0:
                raise(Exception("Non invertable)"))
            rl = 1/self.a
            i1 = -self.b*(rl**2)
            i2 = -self.c*(rl**2)
            i3 = self.d*(rl**2) - 2*self.b*self.c*(rl**3)
        return hyperdual2(rl,i1,i2,i3)
    def get_matrix(self): # returns a matrix representation
        try: # matrix
            a=self.a.shape
            z = np.zeros(a)
            r1 = np.concatenate((self.a,z,z,z),1)
            r2 = np.concatenate((self.b,self.a,z,z),1)
            r3 = np.concatenate((self.c,z,self.a,z),1)
            r4 = np.concatenate((self.d,self.c,self.b,self.a),1)
            A = np.concatenate((r1,r2,r3,r4),0)
        except: # scalar
            A = np.matrix([[self.a,0,0,0],[self.b, self.a,0,0],[self.c, 0, self.a, 0],[self.d,self.c,self.b,self.a]])
        return A
    def get_vector(self): # returns a vector representation
        try: # matrix
            a=self.a.shape
            A = np.concatenate((self.a,self.b,self.c,self.d))
        except: # scalar
            A = np.matrix([[self.a],[self.b],[self.c],[self.d]])
        return A
    def norm(self):
        return np.sqrt(self.a**2)
