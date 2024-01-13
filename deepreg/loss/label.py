"""Provide different loss or metrics classes for labels."""

import tensorflow as tf

from deepreg.constant import EPS
from deepreg.loss.util import MultiScaleMixin, NegativeLossMixin
from deepreg.registry import REGISTRY


class SumSquaredDifference(tf.keras.losses.Loss):
    """
    Actually, mean of squared distance between y_true and y_pred.

    The inconsistent name was for convention.

    y_true and y_pred have to be at least 1d tensor, including batch axis.
    """

    def __init__(
        self,
        name: str = "SumSquaredDifference",
        **kwargs,
    ):
        """
        Init.

        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return mean squared different for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        loss = tf.math.squared_difference(y_true, y_pred)
        loss = self.flatten(loss)
        return tf.reduce_mean(loss, axis=1)


@REGISTRY.register_loss(name="ssd")
class SumSquaredDifferenceLoss(MultiScaleMixin, SumSquaredDifference):
    """Define loss with multi-scaling options."""


class DiceScore(tf.keras.losses.Loss):
    """
    Define dice score.

    The formulation is:

        0. w_fg + w_bg = 1
        1. let y_prod = y_true * y_pred and y_sum  = y_true + y_pred
        2. num = 2 *  (w_fg * y_true * y_pred + w_bg * (1−y_true) * (1−y_pred))
               = 2 *  ((w_fg+w_bg) * y_prod - w_bg * y_sum + w_bg)
               = 2 *  (y_prod - w_bg * y_sum + w_bg)
        3. denom = (w_fg * (y_true + y_pred) + w_bg * (1−y_true + 1−y_pred))
                 = (w_fg-w_bg) * y_sum + 2 * w_bg
                 = (1-2*w_bg) * y_sum + 2 * w_bg
        4. dice score = num / denom

    where num and denom are summed over all axes except the batch axis.

    Reference:
        Sudre, Carole H., et al. "Generalised dice overlap as a deep learning loss
        function for highly unbalanced segmentations." Deep learning in medical image
        analysis and multimodal learning for clinical decision support.
        Springer, Cham, 2017. 240-248.
    """

    def __init__(
        self,
        binary: bool = False,
        background_weight: float = 0.0,
        smooth_nr: float = EPS,
        smooth_dr: float = EPS,
        name: str = "DiceScore",
        **kwargs,
    ):
        """
        Init.

        :param binary: if True, project y_true, y_pred to 0 or 1.
        :param background_weight: weight for background, where y == 0.
        :param smooth_nr: small constant added to numerator in case of zero covariance.
        :param smooth_dr: small constant added to denominator in case of zero variance.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        if background_weight < 0 or background_weight > 1:
            raise ValueError(
                "The background weight for Dice Score must be "
                f"within [0, 1], got {background_weight}."
            )

        self.binary = binary
        self.background_weight = background_weight
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        if self.binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        # (batch, ...) -> (batch, d)
        y_true = self.flatten(y_true)
        y_pred = self.flatten(y_pred)

        # for foreground class
        y_prod = tf.reduce_sum(y_true * y_pred, axis=1)
        y_sum = tf.reduce_sum(y_true + y_pred, axis=1)

        if self.background_weight > 0:
            # generalized
            vol = tf.reduce_sum(tf.ones_like(y_true), axis=1)
            numerator = 2 * (
                y_prod - self.background_weight * y_sum + self.background_weight * vol
            )
            denominator = (
                1 - 2 * self.background_weight
            ) * y_sum + 2 * self.background_weight * vol
        else:
            # foreground only
            numerator = 2 * y_prod
            denominator = y_sum

        return (numerator + self.smooth_nr) / (denominator + self.smooth_dr)

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(
            binary=self.binary,
            background_weight=self.background_weight,
            smooth_nr=self.smooth_nr,
            smooth_dr=self.smooth_dr,
        )
        return config


@REGISTRY.register_loss(name="dice")
class DiceLoss(NegativeLossMixin, MultiScaleMixin, DiceScore):
    """Revert the sign of DiceScore and support multi-scaling options."""


class DiceScoreLayer(tf.keras.layers.Layer):
    def __init__(self, smooth_nr=1e-6, smooth_dr=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

    def call(self, inputs):
        y_true_class, y_pred_class = inputs
        intersection = tf.reduce_sum(y_true_class * y_pred_class, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true_class, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_class, axis=[1, 2, 3])
        dice = (2. * intersection + self.smooth_nr) / (union + self.smooth_dr)

        def blank_fn():
            return tf.fill(tf.shape(intersection), -1.0)

        def real_fn():
            return (2. * intersection + self.smooth_nr) / (union + self.smooth_dr)

        # if union is empty, return -1
        return tf.cond(tf.reduce_all(tf.less(union, 1)), blank_fn, real_fn)


class MultiClassDiceScore(tf.keras.losses.Loss):
    """
    Define dice score.

    The formulation is:

        0. w_fg + w_bg = 1
        1. let y_prod = y_true * y_pred and y_sum  = y_true + y_pred
        2. num = 2 *  (w_fg * y_true * y_pred + w_bg * (1−y_true) * (1−y_pred))
               = 2 *  ((w_fg+w_bg) * y_prod - w_bg * y_sum + w_bg)
               = 2 *  (y_prod - w_bg * y_sum + w_bg)
        3. denom = (w_fg * (y_true + y_pred) + w_bg * (1−y_true + 1−y_pred))
                 = (w_fg-w_bg) * y_sum + 2 * w_bg
                 = (1-2*w_bg) * y_sum + 2 * w_bg
        4. dice score = num / denom

    where num and denom are summed over all axes except the batch axis.

    This score is implemented separately for each class, then averaged.

    Reference:
        Sudre, Carole H., et al. "Generalised dice overlap as a deep learning loss
        function for highly unbalanced segmentations." Deep learning in medical image
        analysis and multimodal learning for clinical decision support.
        Springer, Cham, 2017. 240-248.
    """

    def __init__(
        self,
        binary: bool = False,
        use_non_default_background_weight: bool = False,
        background_weight: float = 0.0,
        smooth_nr: float = EPS,
        smooth_dr: float = EPS,
        max_num_classes: int = 200,
        name: str = "DiceScore",
        **kwargs,
    ):
        """
        Init.

        :param binary: if True, project y_true, y_pred to 0 or 1.
        :param use_non_default_background_weight: whether to weigh background differently than foreground.
        :param background_weight: weight for background, where y == 0.
        :param smooth_nr: small constant added to numerator in case of zero covariance.
        :param smooth_dr: small constant added to denominator in case of zero variance.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        if background_weight < 0 or background_weight > 1:
            raise ValueError(
                "The background weight for Dice Score must be "
                f"within [0, 1], got {background_weight}."
            )

        self.binary = binary
        self.use_non_default_background_weight = use_non_default_background_weight
        self.background_weight = background_weight
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.max_num_classes = max_num_classes
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        if self.binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        num_classes = self.max_num_classes

        class_indices = tf.range(num_classes)

        # y_true starts in integer encoding
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes, axis=-1)
        # y_pred should already be in one-hot encoding due to the warping
        y_pred_one_hot = y_pred# tf.one_hot(y_pred, depth=num_classes, axis=-1)

        dice_layer = DiceScoreLayer(smooth_nr=self.smooth_nr,
                smooth_dr=self.smooth_dr)

        # Function to compute Dice Score for each class
        def compute_dice_for_class(class_index):
            y_true_class = y_true_one_hot[..., class_index]
            y_pred_class = y_pred_one_hot[..., class_index]
            return dice_layer((y_true_class, y_pred_class))

        scores = [compute_dice_for_class(cls) for cls in class_indices]

        scores = tf.stack(scores, axis=-1)

        mask = scores >= 0

        # filter out negative entries corresponding to blank slices
        scores = tf.where(mask, scores, tf.zeros_like(scores))

        scores_sum = tf.reduce_sum(scores, axis=-1)
        num_positive_scores = tf.reduce_sum(tf.cast(mask, tf.float32), axis=-1)
        mean_scores = scores_sum / (num_positive_scores + tf.keras.backend.epsilon())

        if not self.use_non_default_background_weight:
            return tf.reduce_mean(scores, axis=-1)
        
        assert mask[0] > 0, "Background must be present."
        background_scores = scores[..., 0]
        foreground_scores = scores[..., 1:]
        weighted_background_score = tf.reduce_mean(background_scores) * self.background_weight
        weighted_foreground_score = tf.reduce_mean(foreground_scores) * (1 -
                self.background_weight) / (num_positive_scores - 1 +
                        tf.keras.backend.epsilon())
        total_score = weighted_background_score + weighted_foreground_score
        return total_score / (self.background_weight + (1 - self.background_weight) * (num_classes - 1))



    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(
            binary=self.binary,
            background_weight=self.background_weight,
            smooth_nr=self.smooth_nr,
            smooth_dr=self.smooth_dr,
        )
        return config


@REGISTRY.register_loss(name="multiclass_dice")
class MultiClassDiceLoss(NegativeLossMixin, MultiScaleMixin, MultiClassDiceScore):
    """Revert the sign of DiceScore and support multi-scaling options."""


class CrossEntropy(tf.keras.losses.Loss):
    """
    Define weighted cross-entropy.

    The formulation is:
        loss = − w_fg * y_true log(y_pred) - w_bg * (1−y_true) log(1−y_pred)
    """

    def __init__(
        self,
        binary: bool = False,
        background_weight: float = 0.0,
        smooth: float = EPS,
        name: str = "CrossEntropy",
        **kwargs,
    ):
        """
        Init.

        :param binary: if True, project y_true, y_pred to 0 or 1
        :param background_weight: weight for background, where y == 0.
        :param smooth: smooth constant for log.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        if background_weight < 0 or background_weight > 1:
            raise ValueError(
                "The background weight for Cross Entropy must be "
                f"within [0, 1], got {background_weight}."
            )
        self.binary = binary
        self.background_weight = background_weight
        self.smooth = smooth
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        if self.binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        # (batch, ...) -> (batch, d)
        y_true = self.flatten(y_true)
        y_pred = self.flatten(y_pred)

        loss_fg = -tf.reduce_mean(y_true * tf.math.log(y_pred + self.smooth), axis=1)
        if self.background_weight > 0:
            loss_bg = -tf.reduce_mean(
                (1 - y_true) * tf.math.log(1 - y_pred + self.smooth), axis=1
            )
            return (
                1 - self.background_weight
            ) * loss_fg + self.background_weight * loss_bg
        else:
            return loss_fg

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(
            binary=self.binary,
            background_weight=self.background_weight,
            smooth=self.smooth,
        )
        return config


@REGISTRY.register_loss(name="cross-entropy")
class CrossEntropyLoss(MultiScaleMixin, CrossEntropy):
    """Define loss with multi-scaling options."""


class JaccardIndex(DiceScore):
    """
    Define Jaccard index.

    The formulation is:
    1. num = y_true * y_pred
    2. denom = y_true + y_pred - y_true * y_pred
    3. Jaccard index = num / denom

        0. w_fg + w_bg = 1
        1. let y_prod = y_true * y_pred and y_sum  = y_true + y_pred
        2. num = (w_fg * y_true * y_pred + w_bg * (1−y_true) * (1−y_pred))
               = ((w_fg+w_bg) * y_prod - w_bg * y_sum + w_bg)
               = (y_prod - w_bg * y_sum + w_bg)
        3. denom = (w_fg * (y_true + y_pred - y_true * y_pred)
                  + w_bg * (1−y_true + 1−y_pred - (1−y_true) * (1−y_pred)))
                 = w_fg * (y_sum - y_prod) + w_bg * (1-y_prod)
                 = (1-w_bg) * y_sum - y_prod + w_bg
        4. dice score = num / denom
    """

    def __init__(
        self,
        binary: bool = False,
        background_weight: float = 0.0,
        smooth_nr: float = EPS,
        smooth_dr: float = EPS,
        name: str = "JaccardIndex",
        **kwargs,
    ):
        """
        Init.

        :param binary: if True, project y_true, y_pred to 0 or 1.
        :param background_weight: weight for background, where y == 0.
        :param smooth_nr: small constant added to numerator in case of zero covariance.
        :param smooth_dr: small constant added to denominator in case of zero variance.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(
            binary=binary,
            background_weight=background_weight,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            name=name,
            **kwargs,
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        if self.binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        # (batch, ...) -> (batch, d)
        y_true = self.flatten(y_true)
        y_pred = self.flatten(y_pred)

        # for foreground class
        y_prod = tf.reduce_sum(y_true * y_pred, axis=1)
        y_sum = tf.reduce_sum(y_true + y_pred, axis=1)

        if self.background_weight > 0:
            # generalized
            vol = tf.reduce_sum(tf.ones_like(y_true), axis=1)
            numerator = (
                y_prod - self.background_weight * y_sum + self.background_weight * vol
            )
            denominator = (
                (1 - self.background_weight) * y_sum
                - y_prod
                + self.background_weight * vol
            )
        else:
            # foreground only
            numerator = y_prod
            denominator = y_sum - y_prod

        return (numerator + self.smooth_nr) / (denominator + self.smooth_dr)


@REGISTRY.register_loss(name="jaccard")
class JaccardLoss(NegativeLossMixin, MultiScaleMixin, JaccardIndex):
    """Revert the sign of JaccardIndex."""


def compute_centroid(mask: tf.Tensor, grid: tf.Tensor) -> tf.Tensor:
    """
    Calculate the centroid of the mask.
    :param mask: shape = (batch, dim1, dim2, dim3)
    :param grid: shape = (1, dim1, dim2, dim3, 3)
    :return: shape = (batch, 3), batch of vectors denoting
             location of centroids.
    """
    mask = tf.cast(mask, dtype=tf.float32)
    assert len(mask.shape) == 4
    assert len(grid.shape) == 5
    bool_mask = tf.expand_dims(
        tf.cast(mask >= 0.5, dtype=tf.float32), axis=4
    )  # (batch, dim1, dim2, dim3, 1)
    masked_grid = bool_mask * grid  # (batch, dim1, dim2, dim3, 3)
    numerator = tf.reduce_sum(masked_grid, axis=[1, 2, 3])  # (batch, 3)
    denominator = tf.reduce_sum(bool_mask, axis=[1, 2, 3])  # (batch, 1)
    return (numerator + EPS) / (denominator + EPS)  # (batch, 3)


def compute_centroid_distance(
    y_true: tf.Tensor, y_pred: tf.Tensor, grid: tf.Tensor
) -> tf.Tensor:
    """
    Calculate the L2-distance between two tensors' centroids.
    :param y_true: tensor, shape = (batch, dim1, dim2, dim3)
    :param y_pred: tensor, shape = (batch, dim1, dim2, dim3)
    :param grid: tensor, shape = (1, dim1, dim2, dim3, 3)
    :return: shape = (batch,)
    """
    centroid_1 = compute_centroid(mask=y_pred, grid=grid)  # (batch, 3)
    centroid_2 = compute_centroid(mask=y_true, grid=grid)  # (batch, 3)
    return tf.sqrt(tf.reduce_sum((centroid_1 - centroid_2) ** 2, axis=1))


def foreground_proportion(y: tf.Tensor) -> tf.Tensor:
    """
    Calculate the percentage of foreground vs background per 3d volume.
    :param y: shape = (batch, dim1, dim2, dim3), a 3D label tensor
    :return: shape = (batch,)
    """
    y = tf.cast(y, dtype=tf.float32)
    y = tf.cast(y >= 0.5, dtype=tf.float32)
    return tf.reduce_sum(y, axis=[1, 2, 3]) / tf.reduce_sum(
        tf.ones_like(y), axis=[1, 2, 3]
    )
