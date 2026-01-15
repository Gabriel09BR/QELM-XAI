"""
ELM and QELM model implementations.
"""
from .elm import (
    ELM_Sigmoid,
    ELM_Tanh,
    ELM_ReLU,
    ELM_Sine,
    ELM_Tribas,
    ELM_Hardlim
)
from .qelm import (
    QELM_Sigmoid,
    QELM_Tanh,
    QELM_ReLU,
    QELM_Sine,
    QELM_Tribas,
    QELM_Hardlim
)

__all__ = [
    'ELM_Sigmoid',
    'ELM_Tanh',
    'ELM_ReLU',
    'ELM_Sine',
    'ELM_Tribas',
    'ELM_Hardlim',
    'QELM_Sigmoid',
    'QELM_Tanh',
    'QELM_ReLU',
    'QELM_Sine',
    'QELM_Tribas',
    'QELM_Hardlim',
]
