"""
Different metrics used in deep learning algorithms
"""
import inspect
import tensorflow as tf
import tensorflow.keras.backend as K


class MetricsManager(object):
    """
    MetricsManager class contains the logic to use common metrics,
    as well as to extend the class to create new metrics than can be directly parsed from a string.
    It also implements some general metrics
    """
    @classmethod
    def get_metric(cls, metric_name, *args, **kwargs):
        """
        Get a "pointer to a metric function" defined in this class (or any child class), based on a metric name and
        a set of optional parameters
        :param metric_name: str. Name of the metric
        :param args: positional parameters
        :param kwargs: named parameters
        :return: "Pointer" to a function that can be used by Keras
        """
        metrics = dict(inspect.getmembers(cls, inspect.ismethod))
        if metric_name not in metrics:
            raise Exception("Metric '{}' not found".format(metric_name))
        metric = metrics[metric_name]
        param_names = inspect.getargspec(metric).args
        try:
            if param_names != ['cls', 'y_true', 'y_pred']:
                # Parametric metric. Return a call to the function with all the positional and named parameters
                metric = metrics[metric_name](*args, **kwargs)
                # This code is not compatible with Python 3.6
                # param_names = metric.__code__.co_varnames
                # internal_function = metric.__code__.co_name
                # if param_names != ('y_true', 'y_pred'):
                #     warnings.warn("The internal function '{0}' in metric '{1}' does not match the expected signature. "
                #                   "Use the signature {0}(y_true, y_pred) in the internal function to remove this warning" \
                #                   .format(internal_function, metric_name))
            return metric
        except:
            raise Exception("There is not a valid signature for the metric '{0}'. Please make sure that the metric "
                           "(or an internal function of it) matches the signature 'my_function(y_true, y_pred)', ".format(metric_name))

    @classmethod
    def contains_metric(cls, metric_name):
        """
        Indicate if the passed metric belongs to the class
        :param metric_name: str. Metric name
        :return: Boolean
        """
        metrics = dict(inspect.getmembers(cls, inspect.ismethod))
        return metric_name in metrics


    ###########################################################################################################
    # METRICS #
    ###########################################################################################################
    @classmethod
    def dice_coef(cls, y_true, y_pred):
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @classmethod
    def dice_coef_loss(cls, y_true, y_pred):
        return 1.0-cls.dice_coef(y_true, y_pred)

    @classmethod
    def l2(cls, y_true, y_pred):
        """
        L2 loss (squared L2 norm). One value per data point (reduce all the possible dimensions)
        :param y_true: Tensor (tensorflow). Ground truth tensor
        :param y_pred: Tensor (tensorflow). Prediction tensor
        :return: Tensor (tensorflow)
        """
        shape = y_pred.get_shape()
        axis = tuple(range(1, len(shape)))
        result = K.sum((y_true - y_pred) ** 2, axis=axis)
        return result

    @classmethod
    def l1(cls, y_true, y_pred):
        """
        L1 norm  (one value per data point, that will be a 1-D tensor)
        :param y_true: Tensor (tensorflow). Ground truth tensor
        :param y_pred: Tensor (tensorflow). Prediction tensor
        :return: Tensor (tensorflow)
        """
        result = K.sum(K.abs(y_true - y_pred), axis=1)
        return result

    @classmethod
    def focal_loss(cls, gamma=2.0, alpha=1.0):
        """
        Implement focal loss as described in https://arxiv.org/abs/1708.02002
        :param gamma: float
        :param alpha: float
        :return: Tensor (tensorflow)
        """
        gamma = float(gamma)
        alpha = float(alpha)
        def _focal_loss_(y_true, y_pred):
            eps = 1e-12
            y_pred = K.clip(y_pred, eps, 1. - eps)  # improve the stability of the focal loss
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
                (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        return _focal_loss_

    @classmethod
    def categorical_crossentropy_after_softmax(cls, y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred, from_logits=True)



