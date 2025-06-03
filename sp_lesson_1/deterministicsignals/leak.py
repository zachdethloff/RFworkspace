import matplotlib.pyplot as plt
import numpy as np

# Om0 = -2  # frequency of exponential signal
# N = 45  # length of signal


# # DTFT of finite length exponential signal (analytic)
# Om = np.linspace(-np.pi, np.pi, num=1024)
# XN = (
#     np.exp(-1j * (Om - Om0) * (N - 1) / 2)
#     * (np.sin(N * (Om - Om0) / 2))
#     / (np.sin((Om - Om0) / 2))
# )

# # plot spectrum
# plt.figure(figsize=(10, 8))
# plt.plot(Om, abs(XN))
# plt.title(
#     r"Absolute value of the DTFT of a truncated exponential signal "
#     + r"$e^{{j \Omega_0 k}}$ with $\Omega_0=${0:1.2f}".format(Om0)
# )
# plt.xlabel(r"$\Omega$")
# plt.ylabel(r"$|X_N(e^{j \Omega})|$")
# plt.axis([-np.pi, np.pi, -0.5, N + 5])
# plt.grid()

# plt.show()


N = 32  # length of the signal
P = 40  # periodicity of the exponential signal
Om0 = P * (2 * np.pi / N)  # frequency of exponential signal


# truncated exponential signal
k = np.arange(N)
x = np.exp(1j * Om0 * k)

# DTFT of finite length exponential signal (analytic)
Om = np.linspace(0, 2 * np.pi, num=1024)
Xw = (
    np.exp(-1j * (Om - Om0) * (N - 1) / 2)
    * (np.sin(N * (Om - Om0) / 2))
    / (np.sin((Om - Om0) / 2))
)

# DFT of the exponential signal by FFT
X = np.fft.fft(x)
mu = np.arange(N) * 2 * np.pi / N

# plot spectra
plt.figure(figsize=(10, 8))
ax1 = plt.gca()

plt.plot(Om, abs(Xw), label=r"$|X_N(e^{j \Omega})|$")
plt.stem(mu, abs(X), label=r"$|X_N[\mu]|$", basefmt=" ", linefmt="C1", markerfmt="C1o")
plt.ylim([-0.5, N + 5])
plt.title(
    r"Absolute value of the DTFT/DFT of a truncated exponential signal "
    + r"$e^{{j \Omega_0 k}}$ with $\Omega_0=${0:1.2f}".format(Om0),
    y=1.08,
)
plt.legend()

ax1.set_xlabel(r"$\Omega$")
ax1.set_xlim([Om[0], Om[-1]])
ax1.grid()

ax2 = ax1.twiny()
ax2.set_xlim([0, N])
ax2.set_xlabel(r"$\mu$", color="C1")
ax2.tick_params("x", colors="C1")

plt.show()