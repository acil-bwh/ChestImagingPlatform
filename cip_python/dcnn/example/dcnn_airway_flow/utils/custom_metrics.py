import keras.backend as K


def customized_loss(y_true, y_pred):
    d = 0.0019
    squared_diff = K.abs((y_pred - y_true))  # **2.   #bs,7
    mae = K.sum(squared_diff, axis=-1)  # bs,  #keepdims=true

    v1p = y_pred[:, 0]
    v1t = y_true[:, 0]
    v2p = y_pred[:, 1]
    v2t = y_true[:, 1]

    R = d * K.abs(((v1t - v2t) - (v1p - v2p)) / (v1t - v2t))

    return mae + R


def precision_accuracy_loss(d=0.0019, c_prec=2.5):

    def loss_function(y_true, y_pred):
        mae = K.mean(K.sum(K.abs((y_pred - y_true)), axis=-1))

        # v1p = y_pred[:, 0]
        # v1t = y_true[:, 0]
        # v2p = y_pred[:, 1]
        # v2t = y_true[:, 1]
        #
        # R = K.mean(K.abs(((v1t - v2t) - (v1p - v2p)) / (v1t - v2t)))

        m1p = y_pred[:, 2]
        m1t = y_true[:, 2]
        m2p = y_pred[:, 3]
        m2t = y_true[:, 3]

        m1_RE = K.abs((m1t - m1p) / m1t)
        m2_RE = K.abs((m2t - m2p) / m2t)

        R = K.mean(m1_RE - m2_RE)

        accuracy_loss = mae + d * R

        cc = 50
        ll = K.shape(y_true)[-1]
        rr = K.shape(y_true)[0] / cc
        y_true_precision = K.reshape(y_true, (rr, cc, ll))
        y_pred_precision = K.reshape(y_pred, (rr, cc, ll))

        precision_loss = K.sum(K.square(y_true_precision - y_pred_precision), axis=-1) - \
                         K.square(K.mean(y_true_precision - y_pred_precision, axis=-1))
        precision_loss = K.mean(K.mean(precision_loss, axis=-1))

        return accuracy_loss + c_prec * precision_loss

    return loss_function
