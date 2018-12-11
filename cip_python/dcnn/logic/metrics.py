"""
Different metrics used in deep learning algorithms
"""
import os
import warnings
import inspect
import keras.backend as K
import tensorflow as tf


"""
Metrics class contains the logic to use common metrics, as well as extend the class to create new metrics than can 
be directly parsed from a string 
"""
class Metrics(object):
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
                param_names = metric.func_code.co_varnames
                internal_function = metric.func_code.co_name
                if param_names != ('y_true', 'y_pred'):
                    warnings.warn("The internal function '{0}' in metric '{1}' does not match the expected signature. "
                                  "Use the signature {0}(y_true, y_pred) in the internal function to remove this warning" \
                                  .format(internal_function, metric_name))
            return metric
        except:
            raise Exception("There is not a valid signature for the metric '{0}'. Please make sure that the metric "
                           "matches the signature '{0}(y_true, y_pred)'".format(metric_name))

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
        return 1.0-dice_coef(y_true, y_pred)

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

class PrecisionMetrics(Metrics):

    @classmethod
    def dice_coef_prec(cls, num_augmented):
        def _dice_coef_prec_(y_true, y_pred):
            raise NotImplementedError()
        return _dice_coef_prec_
#     def l2_weighted(weights, num_elems_per_group):
#         '''
#         L2 loss grouped and weighted (for instance, 6 coordinates for each structure)
#         Args:
#             weights: numpy array of weights (ex: 6 coords x 10 strs)
#             num_elems_per_group: int. Num elements per group (ex: 6)
#
#         Returns:
#             Loss function
#         '''
#         weights_t = K.constant(weights)
#
#         def _aux_(y_true, y_pred):
#             shape = y_pred.get_shape().as_list()
#             y_true = K.reshape(y_true, (-1, shape[1] // num_elems_per_group, num_elems_per_group))
#             y_pred = K.reshape(y_pred, (-1, shape[1] // num_elems_per_group, num_elems_per_group))
#             diff = (y_true - y_pred) ** 2
#             diffw = diff * weights_t
#             s = K.sum(diffw, axis=(1, 2))
#             return s
#
#         return _aux_
#

#
#     def l1_smooth(y_true, y_pred):
#         l1 = K.abs(y_pred - y_true)
#         # return K.sum(l1, axis=1)
#         result = 0.5 * (l1 ** 2)
#         comp = K.less(l1, 1)
#         if K.backend() == 'tensorflow':
#             result = tf.where(comp, result, l1 - 0.5)
#             result = K.sum(result, axis=1)
#         else:
#             result = K.switch(comp, 0.5 * (result ** 2), result - 0.5)
#
#         return result
#
#     def iou(y_true, y_pred):
#         """
#         Intersection over Union.
#         Each tensor has 4 coordinates:
#         - Top left X
#         - Top left Y
#         - Width
#         - Height
#         Args:
#             y_true: N x 4
#             y_pred: N x 4
#
#         Returns:
#             IoU
#         """
#         x_p = y_pred[:, 0]
#         y_p = y_pred[:, 1]
#         width_p = y_pred[:, 2]
#         height_p = y_pred[:, 3]
#
#         x_t = y_true[:, 0]
#         y_t = y_true[:, 1]
#         width_t = y_true[:, 2]
#         height_t = y_true[:, 3]
#
#         x_union = K.minimum(x_t, x_p)
#         width_union = K.maximum(x_t + width_t, x_p + width_p) - x_union
#         y_union = K.minimum(y_t, y_p)
#         height_union = K.maximum(y_t + height_t, y_p + height_p) - y_union
#
#         x_intersection = K.maximum(x_t, x_p)
#         width_intersection = K.minimum(x_t + width_t, x_p + width_p) - x_intersection
#         y_intersection = K.maximum(y_t, y_p)
#         height_intersection = K.minimum(y_t + height_t, y_p + height_p) - y_intersection
#
#         area_union = width_union * height_union
#         area_intersection = width_intersection * height_intersection
#
#         zeros = K.zeros_like(area_union)
#         # Discard the elements that have a 0 height or width
#         nulls = K.equal(y_true[:, 2] * y_true[:, 3] * y_pred[:, 2] * y_pred[:, 3], zeros)
#         if K.backend() == 'tensorflow':
#             area_union = tf.where(nulls, zeros, area_union)
#             result = tf.where(K.equal(area_union, zeros), zeros, area_intersection / area_union)
#         else:
#             area_union = K.switch(nulls, zeros, area_union)
#             result = K.switch(K.equal(area_union, zeros), zeros, area_intersection / area_union)
#
#         # Set to 0 the boxes that are not overlapped (they will have some negative value)
#         result = K.maximum(result, zeros)
#         return result
#
#     def categorical_accuracy_top_k(k):
#         def _aux_(y_true, y_pred):
#             return kmetrics.top_k_categorical_accuracy(y_true, y_pred, k=2)
#
#         _aux_.__name__ = "categorical_accuracy_top_{}".format(k)
#         return _aux_
#
#     def precision(threshold=0.5):
#         def precision_(y_true, y_pred):
#             """Precision metric.
#             Computes the precision over the whole batch using threshold_value.
#             """
#             threshold_value = threshold
#             # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
#             y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
#             # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
#             true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1), axis=1))
#             # count the predicted positives
#             predicted_positives = K.sum(y_pred, axis=1)
#             # Get the precision ratio
#             precision_ratio = true_positives / (predicted_positives + K.epsilon())
#             return precision_ratio
#
#         return precision_
#
#     def recall(threshold=0.5):
#         def recall_(y_true, y_pred):
#             """Recall metric.
#             Computes the recall over the whole batch using threshold_value.
#             """
#             threshold_value = threshold
#             # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
#             y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
#             # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
#             true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
#             # Compute the number of positive targets.
#             possible_positives = K.sum(K.clip(y_true, 0, 1))
#             recall_ratio = true_positives / (possible_positives + K.epsilon())
#             return recall_ratio
#
#         return recall_
#
#     def f1(threshold=0.5):
#         def f1_(y_true, y_pred):
#             """F1 score metric.
#             Computes the F1 score over the whole batch using threshold_value.
#             """
#             threshold_value = threshold
#             # PRECISION
#             # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
#             y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
#             # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
#             true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1), axis=1))
#             # count the predicted positives
#             predicted_positives = K.sum(y_pred, axis=1)
#             # Get the precision ratio
#             precision = true_positives / (predicted_positives + K.epsilon())
#
#             # RECALL
#             # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
#             # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
#             # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
#             # true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
#             # Compute the number of positive targets.
#             possible_positives = K.sum(K.clip(y_true, 0, 1), axis=1)
#             recall = true_positives / (possible_positives + K.epsilon())
#
#             # F1 score
#             f1_score = 2.0 * precision * recall / (precision + recall + K.epsilon())
#             return f1_score
#
#         return f1_
#
#     def precision_binary(y_true, y_pred):
#         """Precision metric in a binary class
#         """
#         true_positives = K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), K.floatx()))
#         predicted_positives = K.sum(K.argmax(y_pred, axis=1))
#         # Get the precision ratio
#         precision_ratio = true_positives / (K.cast(predicted_positives, K.floatx()) + K.epsilon())
#         return precision_ratio
#
#     def recall_binary(y_true, y_pred):
#         """Recall metric in a binary class
#         """
#         true_positives = K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), K.floatx()))
#         possible_positives = K.sum(y_true[:, 1])
#         # Get the precision ratio
#         recall_ratio = true_positives / (possible_positives + K.epsilon())
#         return recall_ratio
#
#     def f1_binary(y_true, y_pred):
#         """F1 score metric in a binary class
#         """
#         # PRECISION
#         true_positives = K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), K.floatx()))
#         predicted_positives = K.sum(K.argmax(y_pred, axis=1))
#         # Get the precision ratio
#         precision = true_positives / (K.cast(predicted_positives, K.floatx()) + K.epsilon())
#
#         # RECALL
#         true_positives = K.sum(K.cast(K.equal(K.argmax(y_true), K.argmax(y_pred)), K.floatx()))
#         possible_positives = K.sum(y_true[:, 1])
#         # Get the precision ratio
#         recall = true_positives / (possible_positives + K.epsilon())
#
#         # F1 score
#         f1_score = 2.0 * precision * recall / (precision + recall + K.epsilon())
#         return f1_score
#
#     def categorical_crossentropy(y_true, y_pred):
#         """Categorical crossentropy reimplementation to prevent overflows with very low outputs
#
#         # Arguments
#             y_pred:
#             y_true:
#
#         # Returns
#             Output tensor.
#         """
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#         y_pred /= K.sum(y_pred, axis=len(y_pred.get_shape()) - 1, keepdims=True)
#
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#         return - K.sum(y_true * K.log(y_pred), axis=len(y_pred.get_shape()) - 1)
#
#     def categorical_crossentropy_multioutput(y_true, y_pred):
#         """
#         Categorical crossentropy for a multi output (average cross_entropy)
#         Args:
#             y_pred: List of arrays
#             y_true: List of arrays
#
#         Returns:
#             Scalar
#         """
#         epsilon = K.epsilon()
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#         y_pred /= K.sum(y_pred, axis=len(y_pred.get_shape()) - 1, keepdims=True)
#
#         y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#         return - K.sum(y_true * K.log(y_pred), axis=len(y_pred.get_shape()) - 1)
#
#     def sigmoid_cross_entropy(y_true, y_pred):
#         """
#         Sigmoid cross entropy using Keras with logits (not default behavior)
#         Args:
#             y_true:
#             y_pred:
#
#         Returns:
#
#         """
#         return K.binary_crossentropy(y_true, y_pred, from_logits=True)
#
#     def pearson_correlation(x, y):
#         """
#         Pearson correlation for two flattened tensors
#         Args:
#             y_true:
#             y_pred:
#
#         Returns:
#
#         """
#         mean_x = K.mean(x)
#         mean_y = K.mean(y)
#         num = K.sum((x - mean_x) * (y - mean_y))
#         den = K.sqrt(K.sum(K.square(x - mean_x))) * K.sqrt(K.sum(K.square(y - mean_y)))
#         den = K.maximum(den, K.epsilon())  # Prevent division by zero
#         result = num / den
#         return result
#
#
#
# class ParametrizedMetrics(object):

