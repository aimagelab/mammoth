{{ fullname | get_headling_module }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   {% for item in attributes %}
   .. autoattribute:: {{ item }}
      :annotation:
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:
      :undoc-members:
      :show-inheritance:
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}
      
   {% for item in functions %}
   .. autofunction:: {{ item }}
      :members:
      :undoc-members:
      :show-inheritance:
   {%- endfor %}
      
   {% for item in functions %}
   .. autofunction:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   {% for item in exceptions %}
   .. autoexception:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

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
{% for item in modules | reorder_modules %}
   {{ item }}
{%- endfor %}

{% endif %}
{% endblock %}