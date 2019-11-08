from tensorflow import keras as k
import tensorflow as tf


class NetworkKeys:
    NUM_UNITS = "num_units"
    ACTIVATION = "activation"
    L2_REG_FACT = "l2_reg_fact"
    DROP_PROB = "drop_prob"
    BATCH_NORM = "batch_norm"


def build_dense_network(input_dim, output_dim, output_activation, params, with_output_layer=True):

    model = k.models.Sequential()

    activation = params.get(NetworkKeys.ACTIVATION, "relu")
    l2_reg_fact = params.get(NetworkKeys.L2_REG_FACT, 0.0)
    regularizer = k.regularizers.l2(l2_reg_fact) if l2_reg_fact > 0 else None
    drop_prob = params.get(NetworkKeys.DROP_PROB, 0.0)
    batch_norm = params.get(NetworkKeys.BATCH_NORM, False)

    last_dim = input_dim
    for i in range(len(params[NetworkKeys.NUM_UNITS])):
        model.add(k.layers.Dense(units=params[NetworkKeys.NUM_UNITS][i],
                                 kernel_regularizer=regularizer,
                                 input_dim=last_dim))
        if batch_norm:
            model.add(k.layers.BatchNormalization())
        model.add(k.layers.Activation(activation))
        last_dim = params[NetworkKeys.NUM_UNITS][i]

        if drop_prob > 0.0:
            model.add(k.layers.Dropout(rate=drop_prob))
    if with_output_layer:
        model.add(k.layers.Dense(units=output_dim, activation=output_activation))
    return model
