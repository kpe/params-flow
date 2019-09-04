# coding=utf-8
#
# created by kpe on 16.Aug.2019 at 13:37
#

from __future__ import absolute_import, division, print_function


from tensorflow import keras

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class RAdam(keras.optimizers.Adam):
    """
    RAdam implementation of arXiv:1908.03265.

    See RAdam - https://arxiv.org/pdf/1908.03265v1.pdf
    See Adam  - https://arxiv.org/pdf/1412.6980.pdf
    See tf.keras.optimizers.Adam
    See https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lazy_adam.py
    """
    def __init__(self, *args, **kwargs):
        super(RAdam, self).__init__(*args, **kwargs)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)

        ro_inf = 2. / (1. - beta_2_t) - 1.                                  # max len of the approx SMA

        g_t = grad

        # v_t = beta2*v + (1-beta2)*g_t^2
        v_t = beta_2_t * v + (1. - beta_2_t) * math_ops.square(g_t)
        v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)

        # m_t = beta1*m + (1-beta1)*g_t
        m_t = beta_1_t * m + (1. - beta_1_t) * g_t
        m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1. - beta_1_power)

        t = local_step

        ro_t = ro_inf - 2. * t * beta_2_power / (1. - beta_2_power)          # len of the approx. SMA

        def f1():
            v_t_hat = math_ops.sqrt(v_t / (1 - beta_2_power))
            r_t = math_ops.sqrt(((ro_t - 4) * (ro_t - 2) * ro_inf) / ((ro_inf - 4) * (ro_inf - 2) * ro_t))
            return lr_t * r_t * m_t_hat / (v_t_hat + epsilon_t)

        def f2():
            return lr_t * m_t_hat

        with ops.control_dependencies([m_t, v_t]):
            var_delta = control_flow_ops.cond(math_ops.greater(ro_t, 4), true_fn=f1, false_fn=f2)

        var_up = state_ops.assign_sub(var, var_delta, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_up, v_t, m_t])

    def _resource_apply_sparse(self, grad, var, indices):  # pragma: no cover
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        ro_inf = 2. / (1. - beta_2_t) - 1.                                   # max len of the approx SMA

        g_t = grad

        # v_t     = beta2 * v + (1 - beta2) * (g_t * g_t)
        #    v_t  = beta2 * v
        #    v_t +=             (1 - beta2) * g_t^2
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices,
                                             math_ops.square(g_t) * (1 - beta_2_t))

        # m_t      = beta1 * m + (1 - beta1) * g_t
        #     m_t  = beta1 * m
        #     m_t += (1 - beta1) * g_t
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, g_t * (1 - beta_1_t))

        m_t_hat = m_t / (1. - beta_1_power)

        t = local_step

        ro_t = ro_inf - 2. * t * beta_2_power / (1. - beta_2_power)          # len of the approx. SMA

        def f1():
            v_t_hat = math_ops.sqrt(v_t / (1 - beta_2_power))
            r_t = math_ops.sqrt(((ro_t - 4) * (ro_t - 2) * ro_inf) / ((ro_inf - 4) * (ro_inf - 2) * ro_t))
            return lr_t * r_t * m_t_hat / (v_t_hat + epsilon_t)

        def f2():
            return lr_t * m_t_hat

        with ops.control_dependencies([m_t, v_t]):
            var_delta = control_flow_ops.cond(math_ops.greater(ro_t, 4), true_fn=f1, false_fn=f2)

        var_up = state_ops.assign_sub(var, var_delta, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_up, v_t, m_t])

