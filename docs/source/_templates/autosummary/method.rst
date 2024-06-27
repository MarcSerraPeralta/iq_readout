:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ fullname | replace("iq_readout.", "iq_readout::") }}

{# In the fullname, the module name is ambiguous. Using a `::` separator
specifies `iq_readout` as the module name. #}
