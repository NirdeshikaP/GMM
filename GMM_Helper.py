from KMeans import *

N = 1500
K = 3


# Note here x should be 2 X 1
# mu should 2 X 1
# sigma 2 X 2
def find_normal_distribution(x, mu, sigma):
    x_mu = x - mu
    x_mu_transpose = np.transpose(x_mu)
    sigma_inverse = np.linalg.inv(sigma)

    product_1 = np.matmul(x_mu_transpose, sigma_inverse)
    product_2 = np.matmul(product_1, x_mu)

    exponent = -0.5 * product_2

    term_2 = math.exp(exponent)

    determinant_sigma = np.linalg.det(sigma)
    sqrt_det_sigma = math.sqrt(determinant_sigma)

    term_1 = 1.0 / (2 * math.pi * sqrt_det_sigma)

    return term_1 * term_2


# gamma is a matrix. See responsibilities
# List of all n_ks. n_k_s[k-1] gives effective number of points that belong to cluster k.
def find_n_k(gamma):
    n_k_s = []
    for k in range(0, K):
        n_k = 0
        for i in range(0, N):
            n_k += gamma[i][k-1]
        n_k_s[k] = n_k

    return n_k_s


# pi_k
# List of all pi_k. pi_k[k-1] gives mixing co-efficient of cluster k.
def find_mixing_coefficients(n_k_s):
    return [float(n_k)/N for n_k in n_k_s]


# Note: Responsibilities should be N X K matrix.
# Reason: Each point what is responsibility towards each of the K clusters.
# gamma[n-1][k-1] gives Responsibility of n towards cluster k.
def find_responsibilities(data, pi_s, mu, sigma):
    for n in range(0, N):
        sum = []
        sum_k = 0
        for k in range(0, K):
            pi_k = pi_s[k]
            x_n = np.reshape(data[n], newshape=(2, 1))
            mu_k = np.reshape(mu[k], newshape=(2, 1))
            sum_k += pi_k * find_normal_distribution(x=x_n, mu=mu_k, sigma=sigma[k])
        sum.append(sum_k)

    gamma = np.zeros((N, K))

    for n in range(0, N):
        for k in range(0, K):
            x_n = np.reshape(data[n], newshape=(2, 1))
            mu_k = np.reshape(mu[k], newshape=(2, 1))
            numerator = pi_s[k] * find_normal_distribution(x=x_n, mu=mu_k, sigma=sigma[k])
            gamma[n][k] = float(numerator)/sum[n]

    return gamma


# gamma_transpose * data (Array) gives K * 2 matrix => mu for each cluster
# Divide mu[k]/n_k_s[k] to get new_mu[k]
def find_new_mu(data, gamma, n_k_s):
    data = np.array(data)
    gamma_transpose = np.transpose(gamma)

    mu = np.matmul(gamma_transpose, data)
    assert mu.shape == (3, 2)

    for i in range(0, len(mu)):
        mu[i] = float(mu[i])/n_k_s[i]

    return mu


def find_new_sigma(data, mu, gamma, n_k_s):
    data = np.array(data)
    sigma = []
    for k in range(0, K):
        mu_k = mu[k-1]
        mu_k = np.tile(mu_k, (len(data), 1))
        x_mu_k = data - mu_k
        x_mu_k_transpose = np.transpose(data - mu_k)
        sigma_k = np.matmul(x_mu_k_transpose, x_mu_k) # The actual formula is (x-mu)(x-mu)T but according to the arrangement we have, doing the other way around is the right way.

        gamma_transpose = np.transpose(gamma)
        gamma_k = gamma_transpose[k]

        weighted_sigma = [sigma_k * g_k for g_k in gamma_k]
        sum_weighted_sigma = sum(weighted_sigma)

        sigma.append(sum_weighted_sigma/n_k_s[k])

    return sigma


def find_log_likelihood(data, pi_s, mu, sigma):
    ln_sum = 0
    for n in range(0, N):
        sum = 0
        for k in range(0, K):
            pi_k = pi_s[k]
            x_n = np.reshape(data[n], newshape=(2, 1))
            mu_k = np.reshape(mu[k], newshape=(2, 1))
            sum += pi_k * find_normal_distribution(x=x_n, mu=mu_k, sigma=sigma[k])

        ln_sum += np.log(sum)

    return ln_sum


# data = read_data()
# print(np.reshape(data[1], (2, 1)))
# data_transpose = np.transpose(data)
# print(data_transpose[1])
# data = np.array(data)
# mu = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
# print(mu)
# mu_0 = mu[0]
# print(data - mu_0)
#
# sigma = []
# a = np.array([1, 2, 3, 4]).reshape(2, 2)
# b = np.array([1, 2, 3, 4]).reshape(2, 2)
# c = np.array([1, 2, 3, 4]).reshape(2, 2)
#
# sigma.append(a)
# sigma.append(b)
# sigma.append(c)
#
# sigma = np.array(sigma)
# l = [1, 2, 3]
# l = [a * v for v in l]
# print(np.array(l))
# s = sum(l)
# print(s)
#
# print(s.shape)
