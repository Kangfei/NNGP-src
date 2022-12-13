from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
	X = X.ravel()
	mu = mu.ravel()
	uncertainty = 1.96 * np.sqrt(np.diag(cov))

	# confidential interval
	plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)

	plt.plot(X, mu, label='mean')
	for i, sample in enumerate(samples):
		plt.plot(X, sample, lw=1, ls='--', label=f'sample {i + 1}')
	if X_train is not None:
		plt.plot(X_train, Y_train, 'rx')
	plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
	output_path = './gp_example2.pdf'
	plt.savefig(output_path)


X = np.arange(-5, 5, 0.2).reshape(-1, 1)

# 定义噪音参数
noise = 0.4

# 带有噪音的训练数据
X_train = np.arange(-3, 4, 1).reshape(-1, 1)
Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)

rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise ** 2)

# 1D 训练样本
gpr.fit(X_train, Y_train)

# 计算后验预测分布的均值向量与协方差矩阵
mu_s, cov_s = gpr.predict(X, return_cov=True)

# 获得最优核函数参数
l = gpr.kernel_.k2.get_params()['length_scale']
sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])

# 从先验分布（多元高斯分布）中抽取样本点
samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)

print(l, sigma_f)
# 与前面手写的结果比对
#assert (np.isclose(l_opt, l))
#assert (np.isclose(sigma_f_opt, sigma_f))

# 绘制结果
plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples= samples)