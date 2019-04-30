import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PPCA import PPCA


# ---- data generation
def generate_data(N, D, dl, sig2):

    A, _ = np.linalg.qr(np.random.rand(D, dl))

    z = np.random.randn(dl, N)
    Y = np.dot(A, z) + np.sqrt(sig2) * np.random.randn(D, N)

    return Y, A


# ---- data settings
N = 3200
D = 1600
dl = 1
sig2 = 0.3

# ---- experiment settings
Reps = 5
Ms = 20
M_array = np.linspace(0, 0.99, Ms)
R2 = np.zeros((Ms, Reps))

for r in range(Reps):

    print('Rep {0}/{1}'.format(r+1, Reps))

    Y, A = generate_data(N, D, dl, sig2)

    for j, m in enumerate(M_array):

        print('m {0}/{1}'.format(j + 1, Ms))

        ix = np.random.random(Y.shape) < m
        Ynan = Y.copy()
        Ynan[ix] = np.nan

        ppcam = PPCA(Ynan, verbose=False)
        ppcam.fit(dl=1, tol=10 ** -4, max_iter=100)

        A_m, sig2_m, mu_m = ppcam.get_params()

        R2[j, r] = np.cos(PPCA.subspace(A, A_m)) ** 2

        sys.stdout.flush()


# ---- mean and standard error
R2_mean = np.mean(R2, axis=1)
R2_std_err = np.std(R2, axis=1) / np.sqrt(Reps)

# ---- model, equation 2
M_array2 = np.linspace(0, 0.999, 100)
s_eff = 1/sig2 * (1 - M_array2)
alpha = N/D
R2model = (alpha * s_eff**2 - 1) / (s_eff * (1 + alpha * s_eff))
R2model[R2model < 0 ] = 0

# ---- plotting
plt.clf()
plt.plot(M_array2, R2model, c='#ff7f0e')
plt.scatter(M_array, R2_mean, c='#1f77b4', marker='.')
plt.scatter(M_array, R2_mean + R2_std_err, c='#1f77b4', marker='_')
plt.scatter(M_array, R2_mean - R2_std_err, c='#1f77b4', marker='_')
plt.grid()
plt.ylim([0, 1.1])
plt.xlabel('Missing rate')
plt.ylabel('$R^2$')
plt.legend(['$R^2$ model', '$R^2$ sim'])
plt.savefig('task01_R2')

# ---- plotting
plt.clf()
plt.plot(M_array, R2_mean, c='#1f77b4')
plt.fill_between(M_array, R2_mean + R2_std_err, R2_mean - R2_std_err, color='#1f77b4', alpha=0.3)
plt.plot(M_array2, R2model, c='#ff7f0e')
plt.grid()
plt.ylim([0, 1.1])
plt.xlabel('Missing rate')
plt.ylabel('$R^2$')
plt.legend(['$R^2$ simulation', '$R^2$ model'])
plt.savefig('task01_R2')
