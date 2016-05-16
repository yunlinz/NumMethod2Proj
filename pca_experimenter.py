import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
from sklearn.decomposition import SparsePCA
import pickle
import os

mesh = Mesh('sphere.xml')
V = FunctionSpace(mesh, "Lagrange", 2)
u0 = Expression("10*exp(-(pow(x[0], 2) + pow(x[1], 2)) / 0.02)")
u1 = interpolate(u0, V)

Y = pickle.load(open('Y.pickle'))
dt = 0.001
nT = 256
r, c = Y.shape

cols = []
# all columns
cols_al = [i for i in range(256)]
cols.append(cols_al)
# equitime spacing
cols_et = [0, 24,  49,  74,  99, 124,  149, 174, 199,  224, 249]
cols.append(cols_et)
# inverse time spacing
cols_it = [0,  1,   2,   3,   4,   7,   15,  31,  63,  127, 255]
cols.append(cols_it)
# linear time spacing
cols_lt = [0,  127, 191, 223, 239, 247, 251, 252, 253, 254, 255]
cols.append(cols_lt)

errors = np.zeros((nT, len(cols)))
for j in range(len(cols)):
    if not os.path.exists('images_{}'.format(j)):
        os.makedirs('images_{}'.format(j))
    c = cols[j]
    snaps = Y[:, c]
    snaps_mat = np.dot(snaps.T, snaps)
    q, s, qt = np.linalg.svd(snaps_mat)
    nPC = 10
    eigvals = np.zeros(nPC)
    Phi = np.zeros(shape=(r, nPC))
    for i in range(nPC):
        Phi[:, i] = 1/np.sqrt(s[i]) * np.dot(snaps, q[:, i])
        eigvals[i] = s[i]
        u = Function(V)
        u.vector().set_local(Phi[:, i])
        wiz = plot(u, interactive=False, scalarbar=False)
        wiz.write_png('images_{}/basis_{}_{}'.format(j, j, i))
    plt.figure(1)
    plt.semilogy(eigvals / eigvals[0])
    #plt.savefig('data/eigs_{}.png'.format(j))


    # now construct the our time stepping matrices
    Phi_i = Function(V)
    Phi_j = Function(V)

    K = np.zeros((nPC, nPC))
    M = np.zeros((nPC, nPC))

    for l in range(nPC):
        for m in range(l, nPC):
            Phi_i.vector().set_local(Phi[:, l])
            Phi_j.vector().set_local(Phi[:, m])
            # using assemble as a numerical integrator

            integral1 = assemble(Phi_i * Phi_j * dx)
            integral2 = dt * assemble(inner(grad(Phi_i), grad(Phi_j)) * dx)

            # print 'i={},j={},Int={}'.format(i,j,assemble(inner(grad(Phi_i), grad(Phi_j)) * dx))

            K[l, m] = integral1 + integral2
            K[m, l] = integral1 + integral2

            M[m, l] = integral1
            M[l, m] = integral1

    u_init = np.dot(interpolate(u0, V).vector().array(), Phi)
    Y2 = np.zeros((nPC, nT))
    Y2[:, 0] = u_init
    for i in range(1, nT):
        Y2[:, i] = np.linalg.solve(K, np.dot(M, Y2[:, i - 1]).T).T
    plt.figure(j + 2)
    plt.plot(Y2.T)
    plt.xlabel('Iteration')
    plt.ylabel('Coefficient')
    plt.title('Component coefficients')
    plt.legend([str(i) for i in range(nPC)])
    plt.savefig('data/weights_{}.png'.format(j))

    # now map them back to our original basis
    Y3 = np.dot(Phi, Y2)

    u_interp = Function(V)
    Y_diff = np.square(Y - Y3)
    for t in range(nT):
        u_interp.vector().set_local(Y_diff[:, t])
        errors[t, j] = assemble(u_interp * dx)


plt.figure(1)
plt.xlabel('Eigen value')
plt.ylabel('Magnitude')
plt.title('Eigenvalue Magnitudes for Different Runs')
plt.legend(['all columns', 'equitime', 'inv time', '1in time'], loc=3)
plt.savefig('data/eigs.png')

plt.figure(11)
plt.semilogy(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Approximation errors')
plt.autoscale(tight=True)
plt.legend(['all columns', 'equitime', 'inv time', '1in time'])
plt.savefig('data/errors.png')

