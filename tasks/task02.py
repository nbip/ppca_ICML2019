import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PPCA import PPCA
from sklearn.datasets import fetch_olivetti_faces

# ---- experiment settings
Reps = 5
Ms = 20
M_array = np.linspace(0, 0.99, Ms)
R2 = np.zeros((Ms, Reps))

# ---- dataset
dataset = fetch_olivetti_faces(shuffle=False)
faces = dataset.data.T

D, N = faces.shape
dl = 1
d = int(np.sqrt(D))

# ---- centering
faces_centered = faces - faces.mean(axis=1)[:, None]
# ---- scaling
faces_scaled = faces_centered / faces_centered.std(axis=1)[:, None]

# ---- surrogate for the true signal direction
[u, s, v] = np.linalg.svd(np.cov(faces_scaled))
A = u[:, 0][:, None]

# ---- example of the 1st signal direction estimate at a given missing rate
m = 0.5
ix = np.random.rand(D, N) < m
Ynan = faces_scaled.copy()
Ynan[ix] = np.nan
ppcam = PPCA(Ynan, verbose=True)
ppcam.fit(dl=1, tol=10 ** -4, max_iter=10000)
A_m, sig2_m, mu_m = ppcam.get_params()

plt.clf()
plt.subplot(311)
plt.imshow(np.reshape(faces_scaled[:, 0], [d, d]), cmap='gray')
plt.axis('off')
plt.title('Example face')
plt.subplot(312)
plt.imshow(np.reshape(A, [d, d]), cmap='gray')
plt.axis('off')
plt.title('First PC as found by PCA')
plt.subplot(313)
plt.imshow(np.reshape(A_m, [d, d]), cmap='gray')
plt.axis('off')
plt.title('First PC as found by PPCA, m = {0:.2f}'.format(m))
plt.tight_layout()
plt.savefig('task02_face')


# ---- signal-to-noise ratio
noise = np.sum(s[1:]) / (D - 1)
signal = s[0] - noise
snr = signal / noise

for r in range(Reps):

    print('Rep {0}/{1}'.format(r+1, Reps))

    for j, m in enumerate(M_array):

        print('m {0}/{1}'.format(j + 1, Ms))

        # ---- introduce missing
        ix = np.random.rand(D, N) < m
        Ynan = faces_scaled.copy()
        Ynan[ix] = np.nan

        # ---- remove full columns/rows of nan
        nanidsD = np.sum(np.isnan(Ynan), axis=1) < N
        nanidsN = np.sum(np.isnan(Ynan), axis=0) < D

        Ynan = Ynan[:, nanidsN]
        Ynan = Ynan[nanidsD, :]

        # ---- fit PPCA
        ppcam = PPCA(Ynan, verbose=False)
        ppcam.fit(dl=1, tol=10 ** -4, max_iter=10000, method='SVD')  # SVD is only done if there is no missing

        _A_m, sig2_m, mu_m = ppcam.get_params()
        A_m = np.zeros((D, 1))
        A_m[nanidsD, :] = _A_m

        R2[j, r] = np.cos(PPCA.subspace(A, A_m)) ** 2

        sys.stdout.flush()


# ---- model, equation 2
M_array2 = np.linspace(0, 0.999, 100)
s_eff = snr * (1 - M_array2)
alpha = N/D
R2model = (alpha * s_eff**2 - 1) / (s_eff * (1 + alpha * s_eff))
R2model[R2model < 0 ] = 0

# since we are using a surrogate measure for the true A,
# we need to scale our simulation results to the max value of the model
R2_scaled = R2 * R2model[0]

# ---- mean and standard error
R2_mean = np.mean(R2_scaled, axis=1)
R2_std_err = np.std(R2_scaled, axis=1) / np.sqrt(Reps)


# ---- plotting
plt.clf()
plt.plot(M_array2, R2model, c='#ff7f0e')
plt.scatter(M_array, R2_mean, c='#1f77b4', marker='.')
plt.scatter(M_array, R2_mean + R2_std_err, c='#1f77b4', marker='_')
plt.scatter(M_array, R2_mean - R2_std_err, c='#1f77b4', marker='_')
plt.grid()
plt.ylim([0, 1.1])
plt.xlim([0, 1])
plt.xlabel('Missing rate')
plt.ylabel('$R^2$')
plt.legend(['$R^2$ model', '$R^2$ sim'])
plt.savefig('task02_R2')



