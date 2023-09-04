#%% Load in libraries
import numpy as np  
import scipy as sp
from classes.hd2 import hyperdual2

#%% Generate Matrices
b,h = 10,10
L = 150
I = b*h**3/12
A = b*h
# Hyperdual2
E = hyperdual2(30e3,1,0,0)
Rho = hyperdual2(2.7e-9,0,1,0)
K = E*I/L**3 * np.matrix([[12,6*L,-12,6*L],[6*L,4*L**2,-6*L,2*L**2],[-12,-6*L,12,-6*L],[6*L,2*L**2,-6*L,4*L**2]])
M = Rho*A*L/420 * np.matrix([[156,22*L,54,-13*L],[22*L,4*L**2,13*L,-3*L**2],[54,13*L,156,-22*L],[-13*L,-3*L**2,-22*L,4*L**2]])

# Validation
# Real
K_0 = E.a*I/L**3 * np.matrix([[12,6*L,-12,6*L],[6*L,4*L**2,-6*L,2*L**2],[-12,-6*L,12,-6*L],[6*L,2*L**2,-6*L,4*L**2]])
M_0 = Rho.a*A*L/420 * np.matrix([[156,22*L,54,-13*L],[22*L,4*L**2,13*L,-3*L**2],[54,13*L,156,-22*L],[-13*L,-3*L**2,-22*L,4*L**2]])
# partial_E
K_1 = K_0/E.a
M_1 = np.matrix(np.zeros((4,4)))
# partial_rho
K_2 = np.matrix(np.zeros((4,4)))
M_2 = M_0/Rho.a
# Double partial
K_12 = np.matrix(np.zeros((4,4)))
M_12 = np.matrix(np.zeros((4,4)))

max_k = np.max(np.abs([K.a-K_0,K.b-K_1,K.c-K_2,K.d-K_12]))
argmax_k = np.argmax(np.abs([K.a-K_0,K.b-K_1,K.c-K_2,K.d-K_12]))
max_m = np.max(np.abs([M.a-M_0,M.b-M_1,M.c-M_2,M.d-M_12]))
argmax_m = np.argmax(np.abs([M.a-M_0,M.b-M_1,M.c-M_2,M.d-M_12]))

k_arg,m_arg = np.floor((argmax_k+1)/16),np.floor((argmax_m+1)/16)
if k_arg==0:
    k_arg_hm = "Real"
elif k_arg == 1:
    k_arg_hm = "1st NonReal"
elif k_arg == 2:
    k_arg_hm = "2nd NonReal"
elif k_arg == 3:
    k_arg_hm = "3rd NonReal"
else:
    k_arg_hm = "error"
if m_arg==0:
    m_arg_hm = "Real"
elif m_arg == 1:
    m_arg_hm = "1st NonReal"
elif m_arg == 2:
    m_arg_hm = "2nd NonReal"
elif m_arg == 3:
    m_arg_hm = "3rd NonReal"
else:
    m_arg_hm = "error"

print(f'Max K different is {max_k} of {k_arg_hm} component.')
print(f'Max M different is {max_m} of {m_arg_hm} component.')
# %% Perform eigen
val,vect = hyperdual2.eigs(K,M)
# Validation
# Real
val_0,vect_0 = sp.linalg.eig(K_0,M_0)
vect_0,val_0 = np.matrix(vect_0),np.real(val_0)
# mass normalize vectors
a = np.diag(vect_0.T*M_0*vect_0)
vect_0 = vect_0/np.sqrt(a)
nm, sha = len(val_0), vect_0.shape
val_1,val_2,val_12 = np.zeros(nm),np.zeros(nm),np.zeros(nm)
phi_1,phi_2,phi_3 = np.matrix(np.zeros(sha)),np.matrix(np.zeros(sha)),np.matrix(np.zeros(sha))
for i in range(nm):
    # 1st Derivative eigenvalue
    val_1[i] = vect_0[:,i].T * (K_1 - val_0[i] * M_1) * vect_0[:,i]
    val_2[i] = vect_0[:,i].T * (K_2 - val_0[i] * M_2) * vect_0[:,i]
    # 1st Derivative eigenvector
    df1 = K_1 - val_1[i] * M_0 - val_0[i] * M_1
    df2 = K_2 - val_2[i] * M_0 - val_0[i] * M_2
    Finv = np.linalg.inv(K_0-val_0[i]*M_0)
    z1 = -Finv * df1 * vect_0[:,i]
    z2 = -Finv * df2 * vect_0[:,i]
    c1 = float(-0.5 * vect_0[:,i].T * M_1 * vect_0[:,i] - vect_0[:,i].T * M_0 * z1)
    c2 = float(-0.5 * vect_0[:,i].T * M_2 * vect_0[:,i] - vect_0[:,i].T * M_0 * z2)
    phi_1[:,i] = z1 + c1 * vect_0[:,i]
    phi_2[:,i] = z2 + c2 * vect_0[:,i]
    # Cross Derivative eigenvalue
    val_12[i] = vect_0[:,i].T * (K_12 - val_1[i] * M_2 - val_2[i] * M_1 - val_0[i] * M_12) * vect_0[:,i] + vect_0[:,i].T * (df1 * phi_2[:,i] + df2 * phi_1[:,i])


max_v = np.max(np.abs([val.a-val_0,(val.b-val_1)/val.b,(val.c-val_2)/val.c,(val.d-val_12)/val.d]))
argmax_v = np.argmax(np.abs([val.a-val_0,(val.b-val_1)/val.b,(val.c-val_2)/val.c,(val.d-val_12)/val.d]))

argmax_v = np.floor((argmax_v+1)/4)
if argmax_v==0:
    argmax_v_hm = "Real"
elif argmax_v == 1:
    argmax_v_hm = "1st NonReal"
elif argmax_v == 2:
    argmax_v_hm = "2nd NonReal"
elif argmax_v == 3:
    argmax_v_hm = "3rd NonReal"
else:
    argmax_v_hm = "error"

print(f'Max Eigenvalue different is {max_v} of {argmax_v_hm} component.')