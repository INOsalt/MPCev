
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

class StandardizedGP(BaseEstimator, RegressorMixin):
    def __init__(self, kernel=None, n_restarts_optimizer=10, alpha=1e-2, downsample_rate=1):
        """
        初始化标准化高斯过程模型。

        参数:
        - kernel: 核函数 (默认: 常数核 * RBF 核)
        - n_restarts_optimizer: 优化器重启次数
        - alpha: 噪声项
        - downsample_rate: 降采样率，用于减少训练数据规模
        """
        if kernel is None:
            kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=10.0, length_scale_bounds=(1e-1, 1e3))
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.downsample_rate = downsample_rate
        self.scaler = StandardScaler()
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           n_restarts_optimizer=self.n_restarts_optimizer,
                                           alpha=self.alpha)

    def fit(self, X, y):
        """
        训练高斯过程模型。

        参数:
        - X: 输入特征 (二维数组)
        - y: 目标值 (一维数组)

        返回:
        - self: 训练后的模型
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 数据降采样
        if self.downsample_rate > 1:
            indices = np.arange(0, len(X_scaled), step=self.downsample_rate)
            X_scaled = X_scaled[indices]
            y = y[indices]

        # 拟合高斯过程模型
        self.gp.fit(X_scaled, y)
        return self

    def predict(self, X):
        """
        使用高斯过程模型预测。

        参数:
        - X: 输入特征 (二维数组)

        返回:
        - y_pred: 预测值 (一维数组)
        """
        X_scaled = self.scaler.transform(X)  # 标准化特征
        return self.gp.predict(X_scaled)

    def save(self, filename):
        """
        保存模型到文件。

        参数:
        - filename: 保存文件名
        """
        import joblib
        joblib.dump(self, filename)

    @staticmethod
    def load(filename):
        """
        从文件加载模型。

        参数:
        - filename: 保存文件名

        返回:
        - 加载的模型
        """
        import joblib
        return joblib.load(filename)
