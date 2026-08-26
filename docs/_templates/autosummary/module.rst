{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

{% if classes %}

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for item in classes %}
   {{ item }}
{% endfor %}
{% endif %}

{% if functions %}

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for item in functions %}
   {{ item }}
{% endfor %}

{% endif %}