.. include:: ../{{ module }}/index.rst   

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:

{% block modules %}
{% if modules %}


.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
