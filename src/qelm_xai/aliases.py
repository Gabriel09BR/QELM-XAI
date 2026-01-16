from .elm import ELMClassifier

ELM_VARIANTS = [
    "ELMRelu", "ELMSigmoid", "ELMPoly", "ELMLinear",
    "ELMTribas", "ELMHardlim", "ELMSine", "ELMTan", "ELMRadbas",
    "QELMRelu", "QELMSigmoid", "QELMPoly", "QELMLinear",
    "QELMTribas", "QELMHardlim", "QELMSine", "QELMTan", "QELMRadbas",
]

for name in ELM_VARIANTS:
    globals()[name] = type(name, (ELMClassifier,), {})

