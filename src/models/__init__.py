from typing import Dict, Any, Union
from enum import Enum

from .base_module import BaseModule

class AVAILABLE_MODELS(Enum):
    TRANSFORMER_FOR_SCA = 'transformer-for-sca'
    SEQUENTIAL_TRANSFORMER = 'sequential-transformer'
    CAUSAL_VIVIT = 'causal-vivit'
    ARLM = 'arlm'

def load(model_name: Union[str, AVAILABLE_MODELS], config_kwargs: Dict[str, Any]) -> BaseModule:
    if isinstance(model_name, str):
        model_name = AVAILABLE_MODELS(model_name)
    if model_name == AVAILABLE_MODELS.TRANSFORMER_FOR_SCA:
        from .transformer_for_sca import Transformer, TransformerConfig
        config = TransformerConfig(**config_kwargs)
        model = Transformer(config)
    elif model_name == AVAILABLE_MODELS.SEQUENTIAL_TRANSFORMER:
        from .sequential_transformer_image_classifier import Transformer, Config
        config = Config(**config_kwargs)
        model = Transformer(config)
    elif model_name == AVAILABLE_MODELS.CAUSAL_VIVIT:
        from .causal_vivit import Config
        config = Config(**config_kwargs)
        if config.sparse_inputs and isinstance(config.per_frame_patch_count, list):
            from .causal_vivit import DecomposedSparseInputTransformer
            model = DecomposedSparseInputTransformer(config)
        elif config.sparse_inputs:
            from .causal_vivit import SparseInputTransformer
            model = SparseInputTransformer(config)
        else:
            from .causal_vivit import Transformer
            model = Transformer(config)
    elif model_name == AVAILABLE_MODELS.ARLM:
        from .autoregressive_diffusion import ARM, ARMConfig
        config = ARMConfig(**config_kwargs)
        model = ARM(config)
    else:
        assert False
    return model