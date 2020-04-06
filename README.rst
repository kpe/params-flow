params-flow
===========

|Build Status| |Coverage Status| |Version Status| |Python Versions| |Downloads|

`params-flow`_ provides an alternative style for defining your `Keras`_ model
or layer configuration in order to reduce the boilerplate code related to
passing and (de)serializing your model/layer configuration arguments.

`params-flow`_ encourages this:

.. code:: python

   import params_flow as pf

   class MyDenseLayer(pf.Layer):      # using params_flow Layer/Model instead of Keras ones
     class Params(pf.Layer.Params):   # extend one or more base Params configurations
       num_outputs = None             # declare all configuration arguments
       activation = "gelu"            #   provide or override super() defaults
                                      # do not define an __init__()

     def build(self, in_shape):
       self.kernel = self.add_variable("kernel",
                                       [int(in_shape[-1]),
                                        self.params.num_outputs])     # access config arguments


which would be sufficient to pass the right configuration arguments to the
super layer/model, as well as take care of (de)serialization, so you can concentrate
on the ``build()`` or ``call()`` implementations, instead of writing boilerplate
code like this:

.. code:: python

    from tf.keras.layers import Layer

    class MyDenseLayer(Layer):
      def __init__(self,
                   num_outputs,            # put all of the layer configuration in the constructor
                   activation = "gelu",    #     provide defaults
                   **kwargs):              # allow base layer configuration to be passed to super
        self.num_outputs = num_outputs
        self.activation = activation
        super().__init__(**kwargs)

      def build(self, in_shape):
        self.kernel = self.add_variable("kernel",
                                        [int(in_shape[-1]),
                                         self.num_outputs])      # access config arguments

      def get_config(self):                # serialize layer configuration, __init__() is the deserializer
        config = {
          'num_outputs': self.num_outputs,
          'activation': self.activation
        }
        base_config = super().get_config()
        return dict(list(base_config.items())) + list(config.items())

NEWS
----
 - **04.Apr.2020** - refactored to use ``WithParams`` mixin from `kpe/py-params`_. Make
   sure to use ``_construct()`` instead of ``__init__()`` in your ``Layer`` and ``Model`` subclasses.
   **Breaking Change** - ``_construct()`` signature has changed, please update
   your ``Layer`` and ``Model`` subclasses from:

   .. code:: python

       def _construct(self, params: Params):
           ...

   to:

   .. code:: python

       def _construct(self, **kwargs):
           super()._construct(**kwargs)
           params = self.params
           ...

 - **11.Sep.2019** - `LookAhead`_ optimizer wrapper implementation for efficient non eager graph mode execution (TPU) added.
 - **05.Sep.2019** - `LookAhead`_ optimizer implementation as Keras callback added.
 - **04.Sep.2019** - `RAdam`_ optimizer implementation added.

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

``params-flow`` provides a ``Layer`` and ``Model`` base classes that help
reducing common boilerplate code in your custom Keras layers and models.

When subclassing a Keras ``Model`` or ``Layer``, each configuration parameter
has to be provided as an argument in ``__init__()``. Keras relies on both ``__init__()``
and ``get_config()`` to make a model/layer serializable.

While python idiomatic this style of defining your Keras models/layers results
in a lot of boilerplate code. `params-flow`_ provides an alternative by
encapsulating all those ``__init__()`` configuration arguments in a dedicated
``Params`` instance (``Params`` is kind of a "type-safe" python dict -
see `kpe/py-params`_).
The model/layer specific configuration needs to be declared as
a nested ``Model.Params``/``Layer.Params`` subclass, and your model/layer have to
subclass ``params_flow.Model``/``params_flow.Layer`` instead of the Keras ones:

.. code:: python

   class BertEmbeddingsLayer(Layer):
     class Params(PositionEmbeddingLayer.Params):
       vocab_size              = None
       token_type_vocab_size   = 2
       hidden_size             = 768
       use_position_embeddings = True

   class TransformerEncoderLayer(Layer):
     class Params(TransformerSelfAttentionLayer.Params,
                  ProjectionLayer.Params):
       intermediate_size       = 3072
       intermediate_activation = "gelu"



this allows you to declare the model's configuration by simply extending
the ``Params`` of the underlying layers:

.. code:: python

  class BertModel(Model):
    class Params(BertEmbeddingsLayer.Params,
                 TransformerEncoderLayer.Params):
      pass

**N.B.** The two code excerpts above are taken from `kpe/bert-for-tf2`_, so check there
for the details of a non-trivial `params-flow`_ based implementation (of `BERT`_).

Resources
---------

- `kpe/py-params`_  - A "type-safe" dict class for python.
- `kpe/bert-for-tf2`_ - BERT implementation using the TensorFlow 2 Keras API with the help of `params-flow`_ for reducing some of the common Keras boilerplate code needed when passing parameters to custom layers.




.. |Build Status| image:: https://travis-ci.org/kpe/params-flow.svg?branch=master
   :target: https://travis-ci.org/kpe/params-flow
.. |Coverage Status| image:: https://coveralls.io/repos/kpe/params-flow/badge.svg?branch=master
   :target: https://coveralls.io/r/kpe/params-flow
.. |Version Status| image:: https://badge.fury.io/py/params-flow.svg
   :target: https://badge.fury.io/py/params-flow
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/params-flow.svg
.. |Downloads| image:: https://img.shields.io/pypi/dm/params-flow.svg

.. _`kpe/py-params`: https://github.com/kpe/py-params
.. _`kpe/params-flow`: https://github.com/kpe/params-flow
.. _`kpe/bert-for-tf2`: https://github.com/kpe/bert-for-tf2
.. _`params-flow`: https://github.com/kpe/params-flow

.. _`Keras`: https://keras.io
.. _`BERT`: https://github.com/google-research/bert
.. _`RAdam`: https://arxiv.org/abs/1908.03265
.. _`LookAhead`: https://arxiv.org/abs/1907.08610

