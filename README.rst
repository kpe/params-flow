params-flow
===========

|Build Status| |Coverage Status| |Version Status| |Python Versions|

TensorFlow Keras utilities and helpers for building custom layers by reducing boilerplate code.

LICENSE
-------

MIT. See `License File <https://github.com/kpe/params-flow/blob/master/LICENSE.txt>`_.

Install
-------

``params-flow`` is on the Python Package Index (PyPI):

::

    pip install params-flow


Usage
-----

``params-flow`` provides a ``Layer`` class that helps reducing common boilerplate
code when writing custom Keras layers.

Instead of defining your layer parameters in ``__init()``, define them in
a ``Params`` class like this:

.. code:: python

    class MyLayer(params_flow.Layer):
        Params(params_flow.Layer.Params):
           hidden_size = 128
           activation  = "gelu"

After extending the ``params.flow.Layer`` like above,
the base class will take care for serializing your layer configuration, and
will spare you from coding comon keras boilerplate code like:

.. code:: python

    class MyLayer(keras.Layer):
        def __init__(self, hidden_size=128, activation="gelu"):
            super(MyLayer, self).__init__()
            self.hidden_size = hidden_size
            self.activation  = activation
        def get_config(self):
            config = {
            "hidden_size": self.hidden_size,
            "activation":  self.activation,
        }
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


.. |Build Status| image:: https://travis-ci.org/kpe/params-flow.svg?branch=master
   :target: https://travis-ci.org/kpe/params-flow
.. |Coverage Status| image:: https://coveralls.io/repos/kpe/params-flow/badge.svg?branch=master
   :target: https://coveralls.io/r/kpe/params-flow
.. |Version Status| image:: https://badge.fury.io/py/params-flow.svg
   :target: https://badge.fury.io/py/params-flow
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/setuptools.svg