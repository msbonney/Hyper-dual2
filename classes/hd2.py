import numpy as np
import scipy as sp

class hyperdual2:
    order = 2
    def __init__(self,a:float,b:float=0,c:float=0,d:float=0):
        self.a = a
        if isinstance(a,np.matrix):
            sh = a.shape
            if not isinstance(b,np.matrix):
                self.b = np.zeros(sh)
            else:
                self.b = b
            if not isinstance(c,np.matrix):
                self.c = np.zeros(sh)
            else:
                self.c = c
            if not isinstance(d,np.matrix):
                self.d = np.zeros(sh)
            else:
                self.d = d
        else:
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
            A = np.concatenate((self.a,self.b,self.c,self.d))
        except: # scalar
            A = np.matrix([[self.a],[self.b],[self.c],[self.d]])
        return A
    def norm(self):
        return np.sqrt(self.a**2)
    # Static methods
    @staticmethod
    def real_eigs(K:np.matrix,M:np.matrix):
        lamb,phi = sp.linalg.eig(K,M)
        phi,lamb = np.matrix(phi), np.real(lamb) # Real values of eigenvalues only
        # Mass normalize mode shapes
        a = np.diag(phi.T*M*phi)
        phi_n = phi/np.sqrt(a)
        return lamb,phi_n
    @staticmethod
    def eigs(K,M):
        if not isinstance(K,hyperdual2):
            x = hyperdual2(K)
        else:
            x = K
        if not isinstance(M,hyperdual2):
            y = hyperdual2(M)
        else:
            y = M
        # Get real value eigenvalues/eigenvectors
        lamb,phi = hyperdual2.real_eigs(x.a,y.a)
        nm,sha = len(lamb), phi.shape
        i1,i2,i3 = np.zeros(nm),np.zeros(nm),np.zeros(nm)
        phi_1,phi_2,phi_3 = np.matrix(np.zeros(sha)),np.matrix(np.zeros(sha)),np.matrix(np.zeros(sha))
        # Loop through modes
        for i in range(nm):
            # Some constants
            Finv = np.linalg.inv(x.a-lamb[i]*y.a)
            # First Derivative Eigenvalue
            i1[i] = phi[:,i].T * (x.b-lamb[i]*y.b) * phi[:,i]
            i2[i] = phi[:,i].T * (x.c-lamb[i]*y.c) * phi[:,i]
            # First Derivative Eigenvector
            df1 = x.b - i1[i] * y.a - lamb[i] * y.b
            df2 = x.c - i2[i] * y.a - lamb[i] * y.c
            z1 = -Finv * df1 * phi[:,i]
            z2 = -Finv * df2 * phi[:,i]
            c1 = float(-0.5 * phi[:,i].T * y.b * phi[:,i] - phi[:,i].T * y.a * z1)
            c2 = float(-0.5 * phi[:,i].T * y.c * phi[:,i] - phi[:,i].T * y.a * z2)
            phi_1[:,i] = z1 + c1 * phi[:,i]
            phi_2[:,i] = z2 + c2 * phi[:,i]
            # Second Derivative Eigenvalue
            i3[i] = phi[:,i].T * (x.d - i1[i] * y.c - i2[i] * y.b - lamb[i] * y.d) * phi[:,i] + phi[:,i].T * (df1 * phi_2[:,i] + df2 * phi_1[:,i])
            # Second Derivative Eigenvector
            df3 = x.d - i3[i] * y.a - i1[i] * y.c - i2[i] * y.b - lamb[i] * y.d
            z3 = -Finv * (df3 * phi[:,i] + df1 * phi_2[:,i] + df2 * phi_1[:,i])
            c3 = float(-phi[:,i].T * (0.5 * y.d * phi[:,i] + y.b * phi_2[:,i] + y.c * phi_1[:,i] + y.a * z3) - phi_2[:,i].T * y.a * phi_1[:,i])
            phi_3[:,i] = z3 + c3 * phi[:,i]
        # Assemble hyperdual objects
        lamb_hd = hyperdual2(lamb,i1,i2,i3)
        phi_hd = hyperdual2(phi,phi_1,phi_2,phi_3)
        return lamb_hd, phi_hd 