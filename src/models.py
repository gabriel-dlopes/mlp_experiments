import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from collections.abc import Sequence

#build sequential mlp model
def build_model(cfg_model, meta):
    model = models.Sequential()

    #hidden layers
    if isinstance(cfg_model.units, Sequence) and not isinstance(cfg_model.units, (str, bytes)):
        hidden_layers = [int(u) for u in cfg_model.units] #tranforms the elements of the configlist into integers
    else:
        hidden_layers = [int(cfg_model.units)] #tranforms single numbers or strings into python list

    for i, units in enumerate(hidden_layers):
        if i == 0:
            #first hidden layer with input dimension
            model.add(
                layers.Dense(
                    units,
                    activation=cfg_model.activation,
                    input_shape=(meta["input_dim"],),
                    kernel_regularizer=None
                )
            )
        else:
            #subsequent hidden layers
            model.add(
                layers.Dense(
                    units,
                    activation=cfg_model.activation,
                    kernel_regularizer=None
                )
            )

    #output layer
    model.add(
        layers.Dense(
            meta["num_classes"],
            activation=cfg_model.activation_out
        )
    )

    return model

#get l2 regularizer if defined in config
def _get_regularizer(cfg_model):
    l2_lambda = getattr(cfg_model, "l2", None)
    if l2_lambda is None or l2_lambda <= 0:
        return None
    return regularizers.l2(l2_lambda)
