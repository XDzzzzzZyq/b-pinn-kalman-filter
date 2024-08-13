from abc import ABC, abstractmethod
import numpy as np
import torch

import datasets


def get_operator(config):

    if config.inverse.operator in ['inpaint', 'inpaint_rnd']:
        mask_ds = datasets.get_mask_dataset(config)
        operator = InpaintOperator(mask=mask_ds)

    else:
        raise NotImplementedError

    return operator

class LinearOperators(ABC):

    def __init__(self, **kwargs):
        self.params = kwargs
        self.iter = None

        self.next()
        self.A, self.pL, self.T = self._decompose(None)  # A = pL * T

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def __call__(self, x, keep_shape=False):
        pass

    @abstractmethod
    def to_matrix(self, shape):
        pass

    @abstractmethod
    def _decompose(self, shape):
        """A = p(\Gamma)T"""
        """return p(\Gamma), T"""
        pass

    @abstractmethod
    def decompose(self, shape):
        """A = p(\Gamma)T"""
        """return p(\Gamma), T"""
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        pass

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        pass


class ScalerMult(LinearOperators):
    def __call__(self, x, keep_shape=False):
        return self.params['k'] * x

    def to_matrix(self, shape):
        return self.params['k'] * np.diag(np.ones(shape[0] * shape[1]))


class MatrixMult(LinearOperators):
    def __call__(self, x, keep_shape=False):
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

    def __call__(self, x, keep_shape=False):
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

    def decompose(self, shape):
        pass

def bcmm(m, v):
    """batched channelled matrix multiplication"""
    B, C, M, N = m.shape
    v_ = v.reshape(-1, 1, M)
    m_ = m.reshape(-1, M, N)
    return torch.bmm(v_, m_).reshape(B, C, N)

class InpaintOperator(LinearOperators):

    def next(self):
        if self.iter is None:
            self.iter = iter(self.params['mask'])

        try:
            self.mask = next(self.iter)[0].squeeze(0)
        except StopIteration:
            self.iter = iter(self.params['mask'])
            self.mask = next(self.iter)[0].squeeze(0)

    def __call__(self, x, keep_shape=True, invert=False):
        assert self.mask.shape == x.shape
        self.mask = self.mask.to(x.device)

        if keep_shape:
            if invert:
                return (1-self.mask) * x
            else:
                return self.mask * x
        else:
            if invert:
                assert self.mask.ndim == 4
                N, _, A, B = self.mask.shape
                mat = torch.zeros((N, A * B, A * B)).to(self.mask.device)
                for i in range(N):
                    mat[i] = self._get_single_mat(1-self.mask[i])
                L = [self._get_single_decomposed_mat(m)[0] for m in mat.squeeze()]
                L = torch.stack(L)[:,None,:,:]

                return bcmm(L, x)
            else:
                return bcmm(self.pL, x)

    def _get_single_mat(self, mat):
        return torch.diag(mat.flatten()).to(self.mask.device)

    def _get_single_decomposed_mat(self, mat):
        return (mat[torch.where(mat.sum(axis=1) == 1)[0]].T,
                1)

    def _to_matrix(self, shape):
        if self.mask.ndim == 2:
            return self._get_single_mat(self.mask)
        elif self.mask.ndim == 4:
            N, _, A, B = self.mask.shape
            mat = torch.zeros((N, A*B, A*B)).to(self.mask.device)
            for i in range(N):
                mat[i] = self._get_single_mat(self.mask[i])

            return mat[:,None,:,:]
        else:
            raise ValueError('wrong shape')
    def _decompose(self, shape):
        mat = self._to_matrix(shape)
        if self.mask.ndim == 2:
            return (mat,
                    *self._get_single_decomposed_mat(mat))
        elif self.mask.ndim == 4:
            L = [self._get_single_decomposed_mat(m)[0] for m in mat.squeeze()]
            return (mat,
                    torch.stack(L)[:,None,:,:],
                    1)
        else:
            raise ValueError('wrong shape')

    def to_matrix(self, shape):
        return self.A

    def decompose(self, shape):
        return self.A, self.pL, self.T



def observe(x, operator: LinearOperators, sigma=1):
    return operator(x) + np.random.randn(*x.shape) * sigma


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from datasets import load_images_from_folder, trim_images, Binarize

    w = 64
    data = load_images_from_folder("../assets")
    data = trim_images(data, 100, 100, 100+w, 100+w)[0][:, :, 0]
    shape = data.shape

    fig, axe = plt.subplots(nrows=1, ncols=3, figsize=(40, 20))
    axe[0].imshow(data)

    k = 3
    operator = GaussianFilter(shape=(k, k), std=k)
    filtered = operator(data)

    axe[1].imshow(filtered)

    mat = operator.to_matrix(data.shape)
    print(mat.shape, data.flatten().shape)
    transformed = (mat @ data.flatten()).reshape(w-k+1,w-k+1)
    axe[2].imshow(transformed)
    plt.show()

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.Resize(w), transforms.ToTensor(), Binarize(0.5, True)])
    mask_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    mask, _ = mask_data[0]
    mask = mask.squeeze()

    fig, axe = plt.subplots(nrows=1, ncols=3, figsize=(40, 20))
    axe[0].imshow(mask)

    operator = InpaintOperator(mask=mask)
    masked = operator(data)
    axe[1].imshow(masked)

    mat = operator.to_matrix(data.shape)
    transformed = (mat @ data.flatten()).reshape(data.shape)
    axe[2].imshow(transformed)
    plt.show()
    # matrix = operator.to_matrix((data.shape[0], data.shape[1]))


    import configs.inverse.nc_ddpmpp_inpaint as configs
    from datasets import get_mask_dataset, get_dataset

    config = configs.get_config()

    train_ds, _ = get_dataset(config)
    train_iter = iter(train_ds)
    batch, _ = next(train_iter)

    mask_ds = get_mask_dataset(config)
    mask_iter = iter(mask_ds)
    mask, _ = next(mask_iter)

    operator = InpaintOperator(mask=mask.squeeze(0))
    masked_batch = operator(batch)

    from torchvision.utils import make_grid, save_image
    import matplotlib.pyplot as plt
    import numpy as np

    nrow = int(np.sqrt(masked_batch.shape[0]))
    image_grid = make_grid(masked_batch, nrow, padding=0)
    print(image_grid.shape, image_grid.min(), image_grid.max())

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    axe.imshow(image_grid[0])
    plt.show()