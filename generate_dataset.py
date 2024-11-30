import numpy as np
import pickle


def signal(t, f_k):
    t_k = np.random.choice(t, size=K, replace=False)
    s = np.zeros(len(t), dtype=complex)
    for i in range(len(f_k)):
        s += np.sqrt(B_0) * np.sinc(B_0 * (t-t_k[i])) * np.exp(1j * 2 * np.pi * f_k[i] * t)

    return s


def MP(snr):
    t = np.arange(0, T, 1 / B)
    freq = np.linspace(0, B, len(t))
    carry_freqs = np.linspace(B_0/2, B-B_0/2, L)
    f_k = np.random.choice(carry_freqs, size=K, replace=False)

    sampled_vector = signal(t, f_k)
    random_noise = np.random.randn(len(sampled_vector)) + 1j * np.random.randn(len(sampled_vector))
    random_noise_energy = np.sum((np.abs(random_noise) ** 2))
    signal_energy_expected = random_noise_energy * (10 ** (snr / 10.0))
    signal_energy = np.sum((np.abs(sampled_vector) ** 2))
    sampled_vector = sampled_vector * (signal_energy_expected ** 0.5) / (signal_energy ** 0.5)
    total_vector = sampled_vector + random_noise  # AWGN

    label = np.zeros(L)
    for i in range(K):
        index = np.where(carry_freqs == f_k[i])[0]
        label[index] = 1

    # Multicoset Sampling
    c = np.sort(np.random.choice(range(0, L), size=P, replace=False))
    c_t = np.reshape(c, (P, 1))
    y = np.zeros((P, len(total_vector) // L), dtype=complex)
    for i in range(P):
        y[i, :] = total_vector[c[i]:len(total_vector) // L * L:L]
    l = np.reshape(np.linspace(1, L, L), (1, L))
    A = np.exp(-1j * 2 * np.pi * c_t * (l - 1) / L)
    A_inverse = np.linalg.pinv(A)
    f = freq[:len(freq)//L]
    # DTFT * Ts = DFT, so the coeffecient 1/LT will be eleminated
    y_freq = np.exp(-1j * 2 * np.pi * c_t * f * (1/B)) * (np.fft.fft(y) / y.shape[-1])
    x_freq = np.dot(A_inverse, y_freq)
    x_freq_energy = np.sum((np.abs(x_freq) ** 2))
    x_norm = x_freq / (x_freq_energy ** 0.5)

    return x_norm, label


def generate_dataset():
    nvecs_per_key = 1000
    dataset = {}
    labelset = {}
    snr_vals = np.arange(-20, 20, 2)
    for snr in snr_vals:
        labelset[snr] = np.zeros([nvecs_per_key, L], dtype='int16')
        dataset[snr] = np.zeros([nvecs_per_key, L, N, 2], dtype='float64')
        for i in range(0, nvecs_per_key, 1):
            x, label = MP(snr)
            labelset[snr][i, :] = label[:]
            dataset[snr][i, :, :, 0] = np.real(x)
            dataset[snr][i, :, :, 1] = np.imag(x)

    file_path = f'data/K={K},dataset.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)
    print('Dataset has been saved as', file_path)

    file_path = f'data/K={K},labelset.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(labelset, file)
    print('Labelset has been saved as', file_path)

    return dataset, labelset


if __name__ == '__main__':

    B = 320e6
    B_0 = 8e6
    T = 8e-6
    N = 64
    P = 16
    L = 40
    E = 1
    # Modify the K value to generate datasets with different occupancy rates
    K = 12

    generate_dataset()
