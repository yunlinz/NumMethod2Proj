from dolfin import *
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

import dolfin
# Create mesh and define function space

mesh = Mesh('sphere.xml')
V = FunctionSpace(mesh, "Lagrange", 2)
u = TrialFunction(V)
v = TestFunction(V)

u0 = Expression("10*exp(-(pow(x[0], 2) + pow(x[1], 2)) / 0.02)")
u0.t = 0

u1 = interpolate(u0, V)
Tmax = 2
nT = 256
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
    print 'timestep: {}'.format(i)
    T += dt
    b = assemble(L)
    u0.t = T
    solve(A, u.vector(), b)
    u1.assign(u)
    Y[:, i] = u.vector().array()
    # plot(u)

pickle.dump(Y, open('Y.pickle','w'))
pickle.dump(V, open('V.pickle','w'))
pickle.dump(mesh, 'mesh.pickle')
print 'all snaps saved'
exit()
Snap = np.dot(np.transpose(Y),Y)

q, s, qt = np.linalg.svd(Snap)

nPCS = np.array([1,2,3,4,5,10])
MaxErr = np.zeros(6)
print "nPC   Max Error" 
# Try various numbers of snapshots, compare the results
for k in range(0,6):
    nPC = nPCS[k] # only want k PC's
    x_vals = np.arange(0, nPC, 1);
    eigvals = np.zeros(nPC)
    Phi = np.zeros(shape=((u1.vector()).size(),nPC))
    for i in range(0,nPC): 
        Phi[:,i] = 1/np.sqrt(s[i])*np.dot(Y,q[:,i])
        eigvals[i] = s[i]
    #Phi = np.dot(u, np.diag(s))[:, :nPC]
    plt.semilogy(x_vals,eigvals)
    plt.savefig('pc_magnitudes_{}.png'.format(nPC))

    # let's plot the PC's
    pc = Function(V)
    for i in range(nPC):
        pc.vector().set_local(Phi[:,i])
        #pl = plot(pc,interactive=True)
        #pl.write_png('pc_component_{}_{}.png'.format(nPC, i))


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

            #print 'i={},j={},Int={}'.format(i,j,assemble(inner(grad(Phi_i), grad(Phi_j)) * dx))

            K[i,j] = integral1 + integral2
            K[j,i] = integral1 + integral2

            M[i,j] = integral1
            M[j,i] = integral1

    # now to do the time stepping, we just solve K * U^k = M * U^k-1
    u_init = np.dot(interpolate(u0, V).vector().array(),Phi)
    #u_init = np.dot(np.dot(interpolate(u0, V).vector().array(), q),
    #                np.linalg.inv(np.diag(s))).T[:nPC] # map this into PCA space to get our initial U^0


    Y2 = np.zeros((nPC, nT))
    Y2[:, 0] = u_init
    for i in range(1, nT):
        Y2[:, i] = np.linalg.solve(K, np.dot(M, Y2[:, i-1]).T).T


    # now map them back to our original basis
    Y3 = np.dot(Phi, Y2)

    u_interp = Function(V)

    for i in range(nT):
        u_interp.vector().set_local(Y3[:, i])
        if MaxErr[k] <= np.amax(u_interp.vector().array()-Y[:,i]):
           MaxErr[k] = np.amax(u_interp.vector().array()-Y[:,i])
        # plot(u_interp)
        # time.sleep(0.5)
        sp = ""
    print "%3d %1s %8.5e" % (nPC, sp, MaxErr[k])

