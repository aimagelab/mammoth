.. include:: ../{{ module }}/index.rst   

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:

{% block modules %}
{% if modules %}

.. autosummary::
   :recursive:
   :toctree:
   :hidden:
   :template: custom-module-template.rst
   {% for item in modules | reorder_modules %}
   {{ item }}
   {%- endfor %}

.. toctree::
   :hidden:
   {% for item in modules |  reorder_modules %}
   {{ item }}
   {%- endfor %}

{% endif %}
{% endblock %}

