"""
Wrappers for optax optimizers to make them compatible with LMRax.
Added Lion.
"""

from optax import (
    adabelief,
    adafactor,
    adagrad,
    adam,
    adamax,
    adamaxw,
    adamw,
    lamb,
    lars,
    novograd,
    radam,
    rmsprop,
    sgd,
)

from lmrax.optimizers.lion import lion

__OPTIMIZERS__ = {
    "adam": adam,
    "adamw": adamw,
    "adabelief": adabelief,
    "adafactor": adafactor,
    "adagrad": adagrad,
    "adamax": adamax,
    "adamaxw": adamaxw,
    "lamb": lamb,
    "lars": lars,
    "lion": lion,
    "novograd": novograd,
    "radam": radam,
    "rmsprop": rmsprop,
    "sgd": sgd,
}


def get(name):
    return __OPTIMIZERS__[name]
