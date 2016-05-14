

from dolfin import *
import numpy as np
import time
from sklearn import decomposition as decomp
import matplotlib.pyplot as plt

import dolfin
# Create mesh and define function space

mesh = Mesh('sphere.xml')
V = FunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)

u0 = Expression("10*exp(-(pow(x[0], 2) + pow(x[1], 2)) / 0.02)")
u0.t = 0

u1 = interpolate(u0, V)
Tmax = 2
nT = 20
dt = Tmax / nT
dt = 0.001
a = u * v * dx + dt * inner(nabla_grad(u), nabla_grad(v)) * dx
L = u1 * v * dx


A = assemble(a)
# Compute solution
u = Function(V)
T = 0
Y = np.zeros((u.vector().array().shape[0], nT))
for i in range(nT):
    T += dt
    b = assemble(L)
    u0.t = T
    solve(A, u.vector(), b)
    u1.assign(u)
    Y[:, i] = u.vector().array()
    plot(u)

u, s, v = np.linalg.svd(Y, full_matrices=False)

nPC = 5 # only want 5 PC's
Phi = np.dot(u, np.diag(s))[:, :nPC]

# let's plot the PC's
pc = Function(V)
for i in range(nPC):
    pc.vector().set_local(Phi[:,i])
    plot(pc)

# time stepping is not working...


# now construct the our time stepping matrices
Phi_i = Function(V)
Phi_j = Function(V)

K = np.zeros((nPC, nPC))
M = np.zeros((nPC, nPC))


for i in range(nPC):
    for j in range(i, nPC):
        Phi_i.vector().set_local(Phi[:, i])
        Phi_j.vector().set_local(Phi[:, j])
        # using assemble as a numerical integrator

        integral1 = assemble(Phi_i * Phi_j * dx)
        integral2 = dt * assemble(inner(grad(Phi_i), grad(Phi_j)) * dx)

        print 'i={},j={},Int={}'.format(i,j,assemble(inner(grad(Phi_i), grad(Phi_j)) * dx))

        K[i,j] = integral1 + integral2
        K[j,i] = integral1 + integral2

        M[i,j] = integral1
        M[j,i] = integral1

# now to do the time stepping, we just solve U^k = K * U^k-1
u_init = np.dot(np.dot(interpolate(u0, V).vector().array(), u),
                np.linalg.inv(np.diag(s))).T[:nPC] # map this into PCA space to get our initial U^0

Y2 = np.zeros((nPC, nT))
Y2[:, 0] = u_init
for i in range(1, nT):
    Y2[:, i] = np.linalg.solve(K, np.dot(M, Y2[:, i-1]).T).T


# now map them back to our original basis
Y3 = np.dot(Phi, Y2)

u_interp = Function(V)

for i in range(nT):
    u_interp.vector().set_local(Y3[:, i])
    plot(u_interp)
    time.sleep(0.5)
