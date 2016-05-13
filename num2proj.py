

from dolfin import *
import numpy as np
from sklearn import decomposition as decomp
import matplotlib.pyplot as plt

import dolfin
# Create mesh and define function space

mesh = Mesh('sphere.xml')
V = FunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)

u0 = Expression("10*sin(-(pow(x[0]-1, 2) + pow(x[1] + 1, 2)) / 0.02)")
u0.t = 0

u1 = interpolate(u0, V)
Tmax = 2
nT = 100
dt = Tmax / nT
dt = 0.001
a = u * v * dx + dt * inner(nabla_grad(u), nabla_grad(v)) * dx
L = u1 * v * dx


A = assemble(a)
# Compute solution
u = Function(V)
T = 0
Y = np.zeros((u.vector().array().shape[0], nT))
print u.vector().array().shape
for i in range(nT):
    T += dt
    b = assemble(L)
    u0.t = T
    solve(A, u.vector(), b)
    u1.assign(u)
    Y[:, i] = u.vector().array()
    plot(u, interactive=False)

U, S, V = np.linalg.svd(Y)

plt.semilogy(S)
plt.show()

nPC = 5 # only want 5 PC's
Phi = np.dot(U, S)[:, nPC]

# now construct the our time stepping matrices
Phi_i = Function(V)
Phi_j = Function(V)
K = np.zeros((nPC, nPC))
for i in range(nPC):
    for j in range(i, nPC):
        Phi_i.vector().set_local(Phi[:, i])
        Phi_j.vector().set_local(Phi[:, j])
        # using assemble as a numerical integrator
        K[i,j] -= assemble(inner(grad(Phi_i), grad(Phi_j)) * dx)
        k[j,i] -= assemble(inner(grad(Phi_i), grad(Phi_j)) * dx)

K += np.eye(nPC)


# now to do the time stepping, we just solve U^k = K * U^k-1
u_init = np.dot(np.dot(interpolate(u0, V).vector().array(),U),
                np.inv(S)).T # map this into PCA space to get our initial U^0

Y2 = np.zeros((nPC, nT))
Y2[:, 0] = u_init
for i in range(1, nT):
    Y2[:, i] = np.dot(np.inv(K), Y2[:, i - 1])


