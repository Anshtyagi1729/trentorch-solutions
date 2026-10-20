import numpy as np


def mse_gradient(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    input:  shape (batch_size, in_features)
    weight: shape (out_features, in_features)
    bias:   shape (out_features,), or None
    target: shape (batch_size, out_features)

    Returns:
        grad_weight: same shape as weight
        grad_bias: same shape as bias, or None if bias is None
    """
    # TODO: Implement the gradient formulas derived in Theory.
    # Do not use an autograd library -- these are the manual derivatives.
    #mse=pred-target **2 
    # to match the dimension to weight matrix , 
    pred=input@weight.T
    if bias is not None:
        pred=pred+bias
    err=pred-target #here the shape is batch, out but for grad we need to match the in , out of weight
    N=err.size
    grad_w=(2/N)*(err.T@input)
    grad_b=None
    if bias is not None:
        grad_b=(2/N)*(np.sum(err,axis=0))
    return grad_w,grad_b
