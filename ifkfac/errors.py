"""
Custom exception hierarchy for the OlsSM optimizer suite.

All public-facing errors inherit from OlsSMError so callers can catch
the entire family with a single except clause:

    try:
        opt = OlsSMKFAC(model, rank=8, adaptive=True)
    except OlsSMError as e:
        print(f"Optimizer misconfigured: {e}")
"""


class OlsSMError(Exception):
    """Base exception for all OlsSM optimizer errors."""


class ConfigurationError(OlsSMError):
    """Raised when optimizer parameters are invalid or mutually exclusive.

    Examples
    --------
    - ``rank`` and ``adaptive=True`` set simultaneously
    - ``decomp_update_freq < factor_update_freq``
    - ``n_layers`` exceeds the number of linear layers in the model
    """


class DataValidationError(OlsSMError):
    """Raised when input data has wrong shape, dtype, or device.

    Examples
    --------
    - ``target_fn`` output shape does not match the last retrained layer
    - Gram matrix is not square
    - Tensor is on a different device than the model
    """


class NumericalError(OlsSMError):
    """Raised when a numerical failure is detected and cannot be recovered.

    Note: non-fatal NaN/inf in Gram matrices are silently skipped during
    training (the layer update is simply omitted for that step).  This
    exception is reserved for unrecoverable conditions, e.g. a completely
    degenerate model where *all* layers produce NaN.
    """


class CheckpointError(OlsSMError):
    """Raised when a state dict cannot be loaded due to architecture mismatch."""
