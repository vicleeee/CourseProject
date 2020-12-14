import json
import math
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import numpy as np
import os
import ssl
import sys
import scipy
from scipy.special import gammaln
import string

nltk.download('wordnet')
nltk.download('stopwords')
stemmer = nltk.wordnet.WordNetLemmatizer()
stopWords = set(stopwords.words('english'))


num_appearances = []
words_dict = []


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix


def update_alpha(r, mu,alpha, s, delta, sigma, W, sigma_inv):
    N = alpha.shape[0] # number of review
    k = alpha.shape[1]
    def func(alpha, r, mu, s, delta, sigma, W, sigma_inv):
        sum = 0
        for d in range (N):
            sum = -(r - np.dot(alpha[d].transpose(), s[d]))**2/(2*delta) - 1/2 * (alpha[d] - mu).transpose()*sigma_inv*(alpha[d] - mu)
        return sum
    def gradient(alpha, r, mu, s, delta, sigma, W, sigma_inv) :
        new_alpha = np.zeros([N, k])
        for d in range (N):
            new_alpha[d] = - (np.dot(alpha[d].transpose(), s[d]) - r[d])*s[d]/delta - sigma_inv*(alpha[d] - mu)
        return new_alpha

    def alpha_hess (alpha, r, mu, s, delta, sigma, W, sigma_inv) :
        sum = 0
        H = np.zeros([N, k])
        for d in range(N):
            for i in range(k):
                for j in range(k):
                    temp += delta* s[d][i]*(1 - alpha[d][i]) - delta* s[d][j]*(1 - alpha[d][j])
                H[d][i] = alpha[d][i] * temp
        return H

    res = minimize(func, alpha, args=(r, mu, s, delta, sigma, W, sigma_inv), jac=gradient, method='Newton-CG',  hess=alpha_hess,
               options={'xtol': 1e-8, 'disp': True})
    return res.x

def helper(curr_reviews, terms):
  reviewMap = {}
  curr, curr_freq, reviews, ratings = [], [], [], [], [], [], []
  for i in range(len(curr_reviews)):
    for review in curr_reviews[i]['Reviews']:
        arr = ['Cleanliness', 'Location', 'Overall', 'Sleep Quality', 'Service', 'Rooms', 'Value',]
        for word in arr:
          tmp.append(review['Ratings'][word])
        ratings.append(review['Ratings']['Overall'])
        
  vocab = np.array(vocab)[np.argsort(vocab)].tolist()
  cnt = np.array(cnt)[np.argsort(vocab)].tolist()
  words_dict = dict(zip(vocab, range(len(vocab))))
  return vocab, cnt, words_dict, curr, curr_freq, reviews, reviewMap, ratings, tmp

def start(k, V, N):
    mu = np.random.random_sample(k) # prior for \alpha in each review
    beta = np.ones([k,len(words_dict)])
    sigma_inv = np.ones([k,k])
    sigma = np.ones([k,k])
    detla = 1.0
    alpha = np.random.multivariate_normal(mu, sigma, N) # mu as mean and sigma as variance parameters
    s = np.zeros([N, k])
    phi = []
    eta = np.ones([N, k])
    gamma = np.ones([N, k])
    epsilon = np.random.random_sample([k, len(words_dict)])
    epsilon =  normalize(epsilon)
    for d in range(0, N):
        L = len(num_appearances[d])
        phi_r = np.ones([L, k])  / float(k)
        phi.append(phi_r)   
        mu += lmbda[d]
        Sigma += (alpha[d] - mu)**2 + sigmaSquare[d]**2
    for i in range(0, k):
        delta += alpha[d]**2 + sigmaSquare[d]**2
    sigmaSquare = np.ones(N, dtype=np.float)
    return mu, beta, sigma_inv, sigma, detla, alpha, s, phi, eta, gamma,epsilon, sigmaSquare


def update_mu(alpha):
    return np.mean(alpha, axis=0)

def update_sigma(alpha, mu):
    return np.dot((alpha-mu).T, (alpha - mu))/len(alpha)

def update_delta(r, alpha, s, sigma): #(r - alpha^T s_d )^2
    N = alpha.shape[0] # number of review
    k = alpha.shape[1]
    for d in range (N):
        delta += np.square(r - np.dot(alpha.transpose(), s))
        for i in range (k):
            delta += (np.sqrare(alpha[i]) + np.sqrare(s[d][i])) * gammaln(s[d][i]) + sigma[d][i]**2*np.mean(s[d][i], axis = 1)
    return delta / N

def update_beta(alpha, beta, W, delta): #W: reviewFreqDict
    N = alpha.shape[0] # number of review
    k = alpha.shape[1]
    def gradient(alpha, beta, r, delta, W):
        derbeta = beta = np.ones([k,N])
        for d in range(N):

            for i in range(k):
                derbeta[i][item] += np.dot(alpha[d],beta[i].transpose() - r) * alpha[d][i]*W[d][i]
        return derbeta
    def func (alpha, beta, r, delta, W):
        num = 0
        for d in range (N):
            sums = 0
            for i in range(k):
                sums += np.dot(alpha[d][i], beta[i].transpose())*W[d][j]
            num += (r[d] - sums)**2/(2*delta)
        return num
    res = minimize(func, args=(r, alpha, W, delta, beta), method='BFGS', jac=gradient,
               options={'disp': True})
    return res.x.reshape((k, N))

def cal_sigmaSquare(delta, s, M, sigma,sigmaSquare , sigma_inv):
    for d in range (k):
        sigmaSquare[i] = delta/(gammaln(s[i]) + np.mean(s[i]) + delta * sigma_inv[i])
    return sigmaSquare

def update_phi(phi, W, beta, mu, gamma, alpha, sigma, s, r, delta, eta):
    N = alpha.shape[0] # number of review
    k = alpha.shape[1]
    new_phi = []
    for d in range (N):
        L = len(d)
        p = np.zeros([L,k])
        for n in range (L):
            for i in range (k):
            p[n][i] = phi[d][n][i]*(gammaln(eta) - gammaln(np.sum(eta[d,:]) + W[d][i]*np.log(epsilon[d][i] - np.log(phi[d][n][i]))) - np.square(np.dot(alpha.transpose(), s))
        p[n,:] = p[n,:] / np.sum(p[n,:])
        new_phi.append(p)
    return new_phi

def update_eta(gamma, alpha, phi):
    N = alpha.shape[0] # number of review
    k = alpha.shape[1]
    for d in range(0, N):
        eta[d,:] = gamma[d,:] + np.sum(phi, axis=0)
    return eta

def update_gamma(gamma, alpha, eta, W):
    N = alpha.shape[0] # number of review
    k = alpha.shape[1]
    new_gamma = np.ones([N, k])
    for i in range (k):
        new_gamma[:,i] = gammaln(np.sum(gamma, axis = 0)) - gammaln(gamma[:,i])
        for d in range (N):
            new_gamma[d][i] += gammaln(eta[d][i]) - gammaln(np.sum(eta, axis = 0))
    return gamma

def calc_log(r,alpha,beta,W,delta,mu,sigma,word_index):
	n_reviews = alpha.shape[0]
	k = alpha.shape[1]
	p = np.exp(alpha)
	p = p / np.sum(p, axis=1, keepdims=True)
	data_likelihood = 0
    entropy =  - (np.dot(alpha.transpose(), s) - r)**2/ delta - 0.5 * np.dot((alpha - mu).transpose(), (alpha - mu))* sigma_inv - 0.5 * np.log(delta) - 0.5* np.log(len(sigma))
    lmbda= 0
	for i in range(n_reviews):
		data_likelihood += (r[i] - np.dot(p[i], np.sum(beta[:, word_index[i]] * W[i], axis=1)))**2 / (2 * delta)
	    data_likelihood = data_likelihood / n_reviews + math.log(delta)
        lmbda += np.square(r - np.dot(alpha.transpose(), s))
        for j in range(k):
            lmbda += (np.sqrare(alpha[i]) + np.sqrare(s[d][i])) * gammaln(s[d][i]) + sigma[d][i]**2*np.mean(s[d][i], axis = 1)
	alpha_likelihood = np.sum((alpha - mu) * np.dot(alpha - mu, inv(sigma + 1e-3 * np.eye(k)))) + math.log(np.linalg.det(sigma + 1e-3 * np.eye(k)))
	beta_likelihood = np.sum(np.square(beta)) * 1e-3
	return data_likelihood + alpha_likelihood + beta_likelihood - entropy - lmbda

def alpha_inference(alpha, r, s, delta, mu, inv_sigma):
	k = alpha.shape[0]
	def f(alpha, r, s, delta, mu, inv_sigma):
		p = np.exp(alpha)
		p = p / np.sum(p)
		return (r - np.dot(p, s)) ** 2 / (2 * delta) + 0.5 * np.dot(alpha - mu, np.dot(inv_sigma, alpha - mu))
	def g(alpha, r, s, delta, mu, inv_sigma):
		p = np.exp(alpha)
		p = p / np.sum(p)
		dpda = -np.outer(p, p)
		dpda[np.diag_indices_from(dpda)] += p
		return np.dot(np.dot((np.dot(p, s) - r), s), dpda) / delta + np.dot(inv_sigma, alpha - mu)
	res = minimize(f, alpha, args=(r, s, delta, mu, inv_sigma), jac=g, method='L-BFGS-B', options={'disp': False})
	return res.x

def E_step(mu, sigma, delta, beta, alpha, W, r, word_index, sigma_inv, phi, eta, gamma,epsilon, sigmaSquare):
    N = alpha.shape[0]
    k = alpha.shape[1]
    new_alpha = np.zeros(alpha.shape)
    s = np.zeros((N, k))
    alpha = alpha_inference(r, mu, alpha, s, delta, sigma, W, sigma_inv)
    eta = update_eta(gamma, alpha, phi)
    phi = update_phi(phi, W, beta, mu, gamma, alpha, sigma, s, r, delta, eta)
    sigmaSquare = cal_sigmaSquare(delta, s, M, sigma,sigmaSquare , sigma_inv)
    for j in range(N):
        s[i] = np.sum(beta[:, i] * W[i], axis=1) 
    return alpha, s, phi

def M_step(mu, sigma, delta, beta, alpha, W, r, word_index, sigma_inv, s, phi, eta, gamma,epsilon, sigmaSquare):
    N = alpha.shape[0]
    k = alpha.shape[1]

    update_gamma(gamma, alpha, eta, W)
    new_mu = update_mu(alpha)
    new_sigma = update_sigma(alpha, new_mu)
    new_delta = update_delta(r, alpha, s, sigma)
    new_beta = update_beta(alpha, beta, W, delta, r)
    return new_mu, new_beta, new_delta, new_sigma

def runEM(r, W):
    k = 7
    V = len(words_dict)
    N = len(num_appearances)
    mu, beta, sigma_inv, sigma, detla, alpha, s, phi,eta, gamma,epsilon, sigmaSquare= start(V, k)
    converge = False
    iteration = 100
    old = calc_log(r, alpha, beta, W,delta, mu, sigma, words_dict)
    while iteration > 0 or not converge:
        alpha, s,new_phi = E_step(mu, sigma, delta, beta, alpha, W, r, words_dict, sigma_inv)
        mu, beta, delta, sigma = M_step(mu, sigma, delta, beta, alpha, W,r, words_dict, sigma_inv, s)
        loglikelihood = calc_log(r, alpha, beta, W,delta, mu, sigma, words_dict)
        e = np.abs(loglikelihood - old)
        if e < 0.001:
            converge = True
        old = loglikelihood
        iteration -= 1
    return s, alpha

curr_reviews, terms = [], []
for file in os.listdir('result'):
    with open(file) as fp:
        terms.append(file.split('.')[0])
        curr_reviews.append(json.load(fp))

helper(curr_reviews, terms)

for k, v in reviewMap.items():
  overall_rating = np.array(ratings)
  W = curr_freq
  word_index = words_dict
  s = runEM(ratings, W)
