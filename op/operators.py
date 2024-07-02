from abc import ABC, abstractmethod
import numpy as np


class LinearOperators(ABC):

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def operate(self, x):
        pass

    @abstractmethod
    def to_matrix(self, shape):
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        pass

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        pass


class ScalerMult(LinearOperators):
    def operate(self, x):
        return self.params['k'] * x

    def to_matrix(self, shape):
        return self.params['k'] * np.diag(np.ones(shape[0] * shape[1]))


class MatrixMult(LinearOperators):
    def operate(self, x):
        return self.params['matrix'] & x

    def to_matrix(self, shape):
        return self.params['matrix']


class GaussianFilter(LinearOperators):

    def get_kernel(self):
        from scipy.stats import multivariate_normal

        gaus = multivariate_normal([0, 0], self.params['std'] * np.eye(2, 2))

        w, h = self.params['shape']
        xa = np.arange(0, w) - w // 2
        ya = np.arange(0, h) - h // 2
        axis = np.meshgrid(xa, ya)
        axis = np.stack(axis, axis=-1).reshape(w, h, -2)
        kernel = gaus.pdf(axis)
        return kernel / kernel.sum()

    def operate(self, x):
        from scipy import signal

        kernel = self.get_kernel()
        return signal.convolve2d(x, kernel, boundary='symm', mode='same')

    def to_matrix(self, shape):
        from scipy.linalg import convolution_matrix

        kernel = self.get_kernel()
        W, H = shape
        kW, kH = kernel.shape
        mW, mH = W-kW+1, H-kH+1

        #print(mW, mH)

        mat = np.zeros((mW*mH, W*H))
        #print(mat.shape)
        for i in range(mW*mH):    # for each pixel
            y_offset = i %  mH
            x_offset = i // mH
            offset = x_offset*H + y_offset
            #print(f"{i}/{mW * mH}, {x_offset}, {y_offset}")
            for r in range(kW):   # for each column in kernel
                #assert len(kernel[r]) == kH
                #print(r, r*H + offset, r*H + offset + kH)
                mat[i, r*H + offset:r*H + offset + kH] = kernel[r]

        return mat


class InpaintOperator(LinearOperators):

    def operate(self, x):
        assert self.params['mask'].shape == x.shape
        return self.params['mask'] * x

    def to_matrix(self, shape):
        return np.diag(self.params['mask'].flatten())


def observe(x, operator: LinearOperators, sigma=1):
    return operator.operate(x) + np.random.randn(*x.shape) * sigma


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from prepocess import *

    w = 64
    data = load_images_from_folder("../assets")
    data = trim_images(data, 100, 100, 100+w, 100+w)[0][:, :, 0]
    shape = data.shape

    fig, axe = plt.subplots(nrows=1, ncols=3, figsize=(40, 20))
    axe[0].imshow(data)

    k = 3
    operator = GaussianFilter(shape=(k, k), std=k)
    filtered = operator.operate(data)

    axe[1].imshow(filtered)

    mat = operator.to_matrix(data.shape)
    print(mat.shape, data.flatten().shape)
    transformed = (mat @ data.flatten()).reshape(w-k+1,w-k+1)
    axe[2].imshow(transformed)
    plt.show()

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.Resize(w), transforms.ToTensor(), Binarize(0.5, True)])
    mask_data = datasets.MNIST(root='../assets', train=True, download=True, transform=transform)
    mask, _ = mask_data[0]
    mask = mask.squeeze()

    fig, axe = plt.subplots(nrows=1, ncols=3, figsize=(40, 20))
    axe[0].imshow(mask)

    operator = InpaintOperator(mask=mask)
    masked = operator.operate(data)
    axe[1].imshow(masked)

    mat = operator.to_matrix(data.shape)
    transformed = (mat @ data.flatten()).reshape(data.shape)
    axe[2].imshow(transformed)
    plt.show()
    # matrix = operator.to_matrix((data.shape[0], data.shape[1]))
