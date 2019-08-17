from cip_python.dcnn.logic.metrics_manager import MetricsManager
import tensorflow.keras.backend as K


class LowToHighDoseMetricsManager(MetricsManager):
    @classmethod
    def adversarial_loss(cls, y_true, y_pred):
        return -K.mean(K.log(y_pred + 1e-12) * y_true + K.log(1 - y_pred + 1e-12) * (1 - y_true))  # Eq 1 in paper

    @classmethod
    def adversarial_loss_lsgan(cls, y_true, y_pred):
        return K.mean(K.abs(K.square(y_pred - y_true)))
