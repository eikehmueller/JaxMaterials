{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

{% if classes %}

Classes
-------

.. autosummary::
   :toctree: generated

{% for item in classes %}
   {{ item }}
{% endfor %}
{% endif %}

{% if functions %}

Functions
---------

.. autosummary::
   :nosignatures:

{% for item in functions %}
   {{ item }}
{% endfor %}

.. autosummary::
   :toctree: generated
   :hidden:

{% for item in functions %}
   {{ item }}
{% endfor %}


{% endif %}