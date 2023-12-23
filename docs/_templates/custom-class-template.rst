{{ name | escape | underline }}

.. currentmodule:: {{ fullname }}

.. autoclass:: {{ fullname }}
   :show-inheritance:

   .. automethod:: {{ name }}.__init__

   {% block attributes %}
   {% if attributes | has_items(name, module) %}

   .. rubric:: {{ _('Attribute details') }}

   {% for item in attributes | drop_torch_items(name, module) %}
   {%- if not item.startswith('_') %}
   .. autoattribute:: {{ name }}.{{ item }}      

   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}

   {% if methods | has_items(name, module) %}
   .. rubric:: {{ _('Methods') }}
      
   {% for item in methods | drop_torch_items(name, module) %}
   .. automethod:: {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
