import torch as th
import torch.nn as nn
import torch.nn.functional as F
from loss_kernels import *

CUDA_AVALABLE = th.cuda.is_available()
DEVICE = th.device("cuda", 0)

DATETIME_FMT = "%Y-%m-%d_%H-%M-%S"

EPSILON = 1E-9
DEBUG_MODE = False


def triu(X):
    # Sum of strictly upper triangular part
    return th.sum(th.triu(X, diagonal=1))


def _atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return th.where(X < eps, X.new_tensor(eps), X)


def d_cs(A, K, n_clusters):
    """
    Cauchy-Schwarz divergence.

    :param A: Cluster assignment matrix
    :type A:  th.Tensor
    :param K: Kernel matrix
    :type K: th.Tensor
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :return: CS-divergence
    :rtype: th.Tensor
    """
    nom = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
    return d


# ======================================================================================================================
# Loss terms
# ======================================================================================================================

class LossTerm:
    # Names of tensors required for the loss computation
    required_tensors = []

    def __init__(self, *args, **kwargs):
        """
        Base class for a term in the loss function.

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        pass

    def __call__(self, net, cfg, extra):
        raise NotImplementedError()


class DDC1(LossTerm):
    """
    L_1 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __call__(self, net, cfg, extra):
        return d_cs(net.output[0], extra["hidden_kernel"], cfg.n_clusters)


class DDC2(LossTerm):
    """
    L_2 loss from DDC
    """
    def __call__(self, net, cfg, extra):
        n = net.output[0].size(0)
        return 2 / (n * (n - 1)) * triu(net.output[0] @ th.t(net.output[0]))


class DDC2Flipped(LossTerm):
    """
    Flipped version of the L_2 loss from DDC. Used by EAMC
    """

    def __call__(self, net, cfg, extra):
        return 2 / (cfg.n_clusters * (cfg.n_clusters - 1)) * triu(th.t(net.output[0]) @ net.output[0])


class DDC3(LossTerm):
    """
    L_3 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def __init__(self, cfg):
        super().__init__()
        self.eye = th.eye(cfg.n_clusters, device=DEVICE)

    def __call__(self, net, cfg, extra):
        m = th.exp(-cdist(net.output[0], self.eye))
        return d_cs(m, extra["hidden_kernel"], cfg.n_clusters)

class kl_div(LossTerm):
    """
    KL divergence loss to avoid cluster collapse
    """
    def __call__(self, net, cfg, extra):
        q = th.ones(net.output[0].shape[1], device=DEVICE)/cfg.n_clusters
        return F.kl_div(th.mean(net.output[0], dim=0), q, reduction='batchmean')

class zinb1(LossTerm):
    """
    zinb loss for highly-sparse rna data
    """
    def __call__(self, net, cfg, extra):

        x = net.mv_input[1][0]
        mean = net.output[2][0]
        disp = net.output[3][0]
        pi = net.output[4][0]
        scale_factor = net.mv_input[2][0][:, None]
        ridge_lambda=0.0
        eps = 1e-10
        mean = mean * scale_factor
        
        t1 = th.lgamma(disp+eps) + th.lgamma(x+1.0) - th.lgamma(x+disp+eps)
        t2 = (disp+x) * th.log(1.0 + (mean/(disp+eps))) + (x * (th.log(disp+eps) - th.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - th.log(1.0-pi+eps)
        zero_nb = th.pow(disp/(disp+mean+eps), disp)
        zero_case = -th.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = th.where(th.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*th.square(pi)
            result += ridge
        
        result = th.mean(result)
        return result

class zinb2(LossTerm):
    """
    zinb loss for highly-sparse rna data
    """
    def __call__(self, net, cfg, extra):

        x = net.mv_input[1][1]
        mean = net.output[2][1]
        disp = net.output[3][1]
        pi = net.output[4][1]
        scale_factor = net.mv_input[2][1][:, None]
        ridge_lambda=0.0
        eps = 1e-10
        mean = mean * scale_factor
        
        t1 = th.lgamma(disp+eps) + th.lgamma(x+1.0) - th.lgamma(x+disp+eps)
        t2 = (disp+x) * th.log(1.0 + (mean/(disp+eps))) + (x * (th.log(disp+eps) - th.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - th.log(1.0-pi+eps)
        zero_nb = th.pow(disp/(disp+mean+eps), disp)
        zero_case = -th.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = th.where(th.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*th.square(pi)
            result += ridge
        
        result = th.mean(result)
        return result

class cca(LossTerm):
    """
    cca loss to make views highly correlated
    """
    def __call__(self, net, cfg, extra):
        r1 = 1e-6
        r2 = 1e-6
        eps = 1e-9
        
        H1, H2 = net.output[1][0], net.output[1][1]
        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)
        m = H1.size(1)
        
        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * th.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * th.matmul(H1bar,H1bar.t()) + r1 * th.eye(o1, device=DEVICE)
        SigmaHat22 = (1.0 / (m - 1)) * th.matmul(H2bar,H2bar.t()) + r2 * th.eye(o2, device=DEVICE)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = th.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = th.symeig(SigmaHat22, eigenvectors=True)

        # Added to increase stability
        posInd1 = th.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = th.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = th.matmul(th.matmul(V1, th.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = th.matmul(th.matmul(V2, th.diag(D2 ** -0.5)), V2.t())
        Tval = th.matmul(th.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
        
        if cfg.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = th.matmul(Tval.t(), Tval)
            corr = th.trace(th.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = th.matmul(Tval.t(), Tval)
            trace_TT = th.add(trace_TT, (th.eye(trace_TT.shape[0])*r1).to(DEVICE)) # regularization for more stability
            U, V = th.symeig(trace_TT, eigenvectors=True)
            U = th.where(U>eps, U, (th.ones(U.shape).float()*eps).to(DEVICE))
            U = U.topk(cfg.outdim_size)[0]
            corr = th.sum(th.sqrt(U))
        return -corr
    
# ======================================================================================================================
# Extra functions
# ======================================================================================================================

def hidden_kernel(net, cfg):
    return vector_kernel(net.hidden, cfg.rel_sigma)


# ======================================================================================================================
# Loss class
# ======================================================================================================================

class Loss(nn.Module):
    # Possible terms to include in the loss
    TERM_CLASSES = {
        "ddc_1": DDC1,
        "ddc_2": DDC2,
        "ddc_2_flipped": DDC2Flipped,
        "ddc_3": DDC3,
        "kl_div":kl_div,
        "zinb1":zinb1,
        "zinb2":zinb2,
        "cca":cca,
    }
    # Functions to compute the required tensors for the terms.
    EXTRA_FUNCS = {
        "hidden_kernel": hidden_kernel,
    }

    def __init__(self, cfg):
        """
        Implementation of a general loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        self.cfg = cfg

        self.names = cfg.funcs.split("|")
        self.weights = cfg.weights if cfg.weights is not None else len(self.names) * [1]

        self.terms = []
        for term_name in self.names:
            self.terms.append(self.TERM_CLASSES[term_name](cfg))

        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))

    def forward(self, net, ignore_in_total=tuple()):
        extra = {name: self.EXTRA_FUNCS[name](net, self.cfg) for name in self.required_extras_names}
        loss_values = {}
        for name, term, weight in zip(self.names, self.terms, self.weights):
            value = term(net, self.cfg, extra)
            # If we got a dict, add each term from the dict with "name/" as the scope.
            if isinstance(value, dict):
                for key, _value in value.items():
                    loss_values[f"{name}/{key}"] = weight * _value
            # Otherwise, just add the value to the dict directly
            else:
                loss_values[name] = weight * value

        loss_values["tot"] = sum([loss_values[k] for k in loss_values.keys() if k not in ignore_in_total])
        
        return loss_values