"""Provide regularization functions and classes for ddf."""
from typing import Callable, Tuple

import tensorflow as tf

from deepreg.registry import REGISTRY
from deepreg.model import layer_util


def gradient_dx(fx: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on x-axis of a 3D tensor using central finite difference.

    It moves the tensor along axis 1 to calculate the approximate gradient, the x axis,
    dx[i] = (x[i+1] - x[i-1]) / 2.

    :param fx: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fx[:, 2:, 1:-1, 1:-1] - fx[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(fy: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on y-axis of a 3D tensor using central finite difference.

    It moves the tensor along axis 2 to calculate the approximate gradient, the y axis,
    dy[i] = (y[i+1] - y[i-1]) / 2.

    :param fy: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fy[:, 1:-1, 2:, 1:-1] - fy[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(fz: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on z-axis of a 3D tensor using central finite difference.

    It moves the tensor along axis 3 to calculate the approximate gradient, the z axis,
    dz[i] = (z[i+1] - z[i-1]) / 2.

    :param fz: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fz[:, 1:-1, 1:-1, 2:] - fz[:, 1:-1, 1:-1, :-2]) / 2


def gradient_dxyz(fxyz: tf.Tensor, fn: Callable) -> tf.Tensor:
    """
    Calculate gradients on x,y,z-axis of a tensor using central finite difference.

    The gradients are calculated along x, y, z separately then stacked together.

    :param fxyz: shape = (..., 3)
    :param fn: function to call
    :return: shape = (..., 3)
    """
    return tf.stack([fn(fxyz[..., i]) for i in [0, 1, 2]], axis=4)

def stable_f(x, min_value=1e-6):
    """
    Perform the operation f(x) = x + 1/x in a numerically stable way.

    This function is intended to penalize growing and shrinking equally.

    :param x: Input tensor.
    :param min_value: The minimum value to which x will be clamped.
    :return: The result of the operation.
    """
    x_clamped = tf.clip_by_value(x, min_value, tf.float32.max)
    return x_clamped + 1.0 / x_clamped

@REGISTRY.register_loss(name="gradient")
class GradientNorm(tf.keras.layers.Layer):
    """
    Calculate the L1/L2 norm of ddf using central finite difference.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, l1: bool = False, name: str = "GradientNorm", **kwargs):
        """
        Init.

        :param l1: bool true if calculate L1 norm, otherwise L2 norm
        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)
        self.l1 = l1

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        dfdx = gradient_dxyz(ddf, gradient_dx)
        dfdy = gradient_dxyz(ddf, gradient_dy)
        dfdz = gradient_dxyz(ddf, gradient_dz)
        if self.l1:
            norms = tf.abs(dfdx) + tf.abs(dfdy) + tf.abs(dfdz)
        else:
            norms = dfdx ** 2 + dfdy ** 2 + dfdz ** 2
        return tf.reduce_mean(norms, axis=[1, 2, 3, 4])

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["l1"] = self.l1
        return config


@REGISTRY.register_loss(name="diff")
class DifferenceNorm(tf.keras.layers.Layer):
    """
    Calculate the average displacement of a pixel in the image, using taxicab metric.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, l1: bool = False, name: str = "DifferenceNorm", **kwargs):
        """
        Init.

        :param l1: bool true if calculate L1 norm, otherwise L2 norm
        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)
        self.l1 = l1


    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["l1"] = self.l1
        return config

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        if self.l1:
            norms = tf.abs(ddf)
        else:
            norms = ddf ** 2
        return tf.reduce_mean(norms, axis=[1, 2, 3, 4])

@REGISTRY.register_loss(name="axisdiff")
class AxisDifferenceNorm(tf.keras.layers.Layer):
    """
    Calculate the average displacement of a pixel in the image, using taxicab metric.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, l1: bool = False, axis: int = 2, name: str = "DifferenceNorm", **kwargs):
        """
        Init.

        :param l1: bool true if calculate L1 norm, otherwise L2 norm
        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)
        self.axis = axis
        self.l1 = l1


    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["l1"] = self.l1
        return config

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        axis = self.axis
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        if self.l1:
            norms = tf.abs(ddf[:,:,:,:,axis:(axis+1)])
        else:
            norms = ddf[:,:,:,:,axis:(axis+1)] ** 2
        return tf.reduce_mean(norms, axis=[1, 2, 3, 4])
    

@REGISTRY.register_loss(name="bending")
class BendingEnergy(tf.keras.layers.Layer):
    """
    Calculate the bending energy of ddf using central finite difference.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, name: str = "BendingEnergy", **kwargs):
        """
        Init.

        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        dfdx = gradient_dxyz(ddf, gradient_dx)
        dfdy = gradient_dxyz(ddf, gradient_dy)
        dfdz = gradient_dxyz(ddf, gradient_dz)

        # second order gradient
        # (batch, m_dim1-4, m_dim2-4, m_dim3-4, 3)
        dfdxx = gradient_dxyz(dfdx, gradient_dx)
        dfdyy = gradient_dxyz(dfdy, gradient_dy)
        dfdzz = gradient_dxyz(dfdz, gradient_dz)
        dfdxy = gradient_dxyz(dfdx, gradient_dy)
        dfdyz = gradient_dxyz(dfdy, gradient_dz)
        dfdxz = gradient_dxyz(dfdx, gradient_dz)

        # (dx + dy + dz) ** 2 = dxx + dyy + dzz + 2*(dxy + dyz + dzx)
        energy = dfdxx ** 2 + dfdyy ** 2 + dfdzz ** 2
        energy += 2 * dfdxy ** 2 + 2 * dfdxz ** 2 + 2 * dfdyz ** 2
        return tf.reduce_mean(energy, axis=[1, 2, 3, 4])

@REGISTRY.register_loss(name="nonrigid")
class NonRigidPenalty(tf.keras.layers.Layer):
    """
    Calculate the L1/L2 norm of ddf using central finite difference.

    Take difference between the norm and the norm of a reference grid to penalize any non-rigid transformation.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, img_size: Tuple[int, int, int] = (0, 0, 0), l1: bool = False, name: str = "NonRigidPenalty", **kwargs):
        """
        Init.

        :param img_size: size of the 3d images, for initializing reference grid
        :param l1: bool true if calculate L1 norm, otherwise L2 norm
        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)
        self.l1 = l1

        # Assert that img_size has been changed from the default value
        assert img_size != (0, 0, 0), "img_size must be set to a value other than (0, 0, 0)"

        self.img_size = img_size
        grid_ref = tf.expand_dims(layer_util.get_reference_grid(grid_size=self.img_size), axis=0)
        self.ddf_ref = -grid_ref

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        dfdx = gradient_dxyz(ddf - self.ddf_ref, gradient_dx)
        dfdy = gradient_dxyz(ddf - self.ddf_ref, gradient_dy)
        dfdz = gradient_dxyz(ddf - self.ddf_ref, gradient_dz)
        if self.l1:
            norms = tf.abs(stable_f(tf.abs(dfdx) + tf.abs(dfdy) + tf.abs(dfdz)) - 2.0)
        else:
            norms = tf.abs(stable_f(dfdx ** 2 + dfdy ** 2 + dfdz ** 2) - 2.0)
        return tf.reduce_mean(norms, axis=[1, 2, 3, 4])

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["l1"] = self.l1
        return config


@REGISTRY.register_loss(name="hybrid")
class HybridNorm(tf.keras.layers.Layer):

    def __init__(self,
        hybrid_weight: dict = {"nonrigid": 0.02,
                        "gradient": 0.02,
                        "diff": 0.005,
                        "axisdiff": 0.001},
        l1: bool = False,
        axis: int = 2,
        name: str = "HybridNorm",
        img_size: Tuple[int, int, int] = (0, 0, 0),
        **kwargs):

        """
        Init.

        :param l1: bool true if calculate L1 norm, otherwise L2 norm
        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)
        self.axis = axis
        self.l1 = l1
        self.nonrigid_weight = hybrid_weight["nonrigid"]
        self.gradientNorm_weight = hybrid_weight["gradient"]
        self.differenceNorm_weight = hybrid_weight["diff"]
        self.axisdiffNorm_weight = hybrid_weight["axisdiff"]
        self.img_size = img_size
        grid_ref = tf.expand_dims(layer_util.get_reference_grid(grid_size=self.img_size), axis=0)
        self.ddf_ref = -grid_ref

        assert img_size != (0, 0, 0), "img_size must be set to a value other than (0, 0, 0)"

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        assert len(inputs.shape) == 5
        ddf = inputs
        # compute the nonrigid penalty
        if self.nonrigid_weight > 0:
            dfdx_nonrigid = gradient_dxyz(ddf - self.ddf_ref, gradient_dx)
            dfdy_nonrigid = gradient_dxyz(ddf - self.ddf_ref, gradient_dy)
            dfdz_nonrigid = gradient_dxyz(ddf - self.ddf_ref, gradient_dz)

            if self.l1:
                nonrigid_norms = tf.abs(
                        stable_f(tf.abs(dfdx_nonrigid) + \
                            tf.abs(dfdy_nonrigid) + tf.abs(dfdz_nonrigid)) - 2.0)
            else:
                nonrigid_norms = tf.abs(stable_f(dfdx_nonrigid ** 2 + \
                    dfdy_nonrigid ** 2 + dfdz_nonrigid ** 2) - 2.0)
            nonrigid_norm = tf.reduce_mean(nonrigid_norms, axis=[1, 2, 3, 4])
        else:
            nonrigid_norm = tf.zeros([tf.shape(ddf)[0]], dtype=ddf.dtype)


        if self.gradientNorm_weight > 0:
            # compute the gradient norm
            dfdx = gradient_dxyz(ddf, gradient_dx)
            dfdy = gradient_dxyz(ddf, gradient_dy)
            dfdz = gradient_dxyz(ddf, gradient_dz)
            if self.l1:
                gradient_norms = tf.abs(dfdx) + tf.abs(dfdy) + tf.abs(dfdz)
            else:
                gradient_norms = dfdx ** 2 + dfdy ** 2 + dfdz ** 2
            gradient_norm = tf.reduce_mean(gradient_norms, axis=[1, 2, 3, 4])
        else:
            gradient_norm = tf.zeros([tf.shape(ddf)[0]], dtype=ddf.dtype)

        # compute the axis difference norm
        if self.axisdiffNorm_weight > 0:
            if self.l1:
                axisdiff_norms = tf.abs(ddf[:,:,:,:,self.axis:(self.axis+1)])
            else:
                axisdiff_norms = ddf[:,:,:,:,self.axis:(self.axis+1)] ** 2
            axisdiff_norm = tf.reduce_mean(axisdiff_norms, axis=[1, 2, 3, 4])
        else:
            axisdiff_norm = tf.zeros([tf.shape(ddf)[0]], dtype=ddf.dtype)

        if self.differenceNorm_weight > 0:
            if self.l1:
                diff_norms = tf.abs(ddf)
            else:
                diff_norms = ddf ** 2
            diff_norm = tf.reduce_mean(diff_norms, axis=[1, 2, 3, 4])
        else:
            diff_norm = tf.zeros([tf.shape(ddf)[0]], dtype=ddf.dtype)
    
        return self.nonrigid_weight * nonrigid_norm + \
            self.gradientNorm_weight * gradient_norm + \
            self.axisdiffNorm_weight * axisdiff_norm + \
            self.differenceNorm_weight * diff_norm