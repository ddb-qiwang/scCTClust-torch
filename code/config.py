# configs
from pydantic import BaseModel
from typing import Tuple, List, Union, Optional
from typing_extensions import Literal

class Config(BaseModel):
    @property
    def class_name(self):
        return self.__class__.__name__
    
class encoder_cfg(Config):
    # Layer dimensions
    Layer: Tuple[Union[int, str], ...] = (2009,256,64,32)
    # Activation function
    activation: Union[str, None, List[Union[None, str]], Tuple[Union[None, str], ...]] = "relu"

class Fusion_config(Config):
    # Fusion method. "mean" constant weights = 1/V. "weighted_mean": Weighted average with learned weights.
    method: Literal["mean", "weighted_mean"]
    # Number of views
    n_views: int
        
class mvae_cfg(Config):
    # config of RNA autoencoder
    rna_ae_cfg = encoder_cfg(Layer=(2009,256,64,32), activation='relu')
    # config of protein autoencoder
    prt_ae_cfg = encoder_cfg(Layer=(10,32), activation='relu')

class DDC_config(Config):
    # Number of clusters
    n_clusters: int = None
    # Number of units in the first fully connected layer
    n_hidden: int = 32
    # Use batch norm after the first fully connected layer?
    use_bn: bool = False
    # If direct or not
    direct: bool = True

class Loss_config(Config):
    # Number of views
    n_views: int = 2
    # Number of clusters
    n_clusters: int = None
    # Terms to use in the loss, separated by '|'. E.g. "ddc_1|ddc_2|ddc_3|kl_div|zinb|cca" for the DDC clustering loss
    funcs: str
    # Optional weights for the loss terms. Set to None to have all weights equal to 1.
    weights: Tuple[Union[float, int], ...] = None
    # Multiplication factor for the sigma hyperparameter
    rel_sigma = 0.15
    # cca para
    use_all_singular_values: bool = False
    # cca para
    outdim_size: int = 32

class Optimizer_config(Config):
    # Base learning rate
    learning_rate: float = 0.0001
    # Max gradient norm for gradient clipping.
    clip_norm: float = 5.0
    # Step size for the learning rate scheduler. None disables the scheduler.
    scheduler_step_size: int = None
    # Multiplication factor for the learning rate scheduler
    scheduler_gamma: float = 0.1
        
class CTClust_cfg(Config):
    # mvae config
    multiview_encoders_config = mvae_cfg()
    # fusion config
    fusion_cfg = Fusion_config(method='weighted_mean', n_views=2)
    # ddc config
    cm_config = DDC_config(n_clusters=6,n_hidden=32,use_bn=False,direct=True)
    # loss config
    loss_cfg = Loss_config(n_views=2,n_clusters=6,funcs="ddc_1|ddc_2|ddc_3|zinb",weights=[1.0,1.0,1.0,1.0,0.02],rel_sigma=0.15)
    # optimizer config
    opt_cfg = Optimizer_config(learning_rate=0.0001)