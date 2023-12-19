Function: {{ name }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ fullname }}

   {% for item in functions %}
   {%- if not item.startswith('_') %}
   ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}

   {% block parameters %}
   {% if parameters %}
   .. rubric:: {{ _('Parameters') }}

   {% for param in parameters %}
   {{ param.name }} : {{ param.type }}
       {{ param.desc }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block returns %}
   {% if returns %}
   .. rubric:: {{ _('Returns') }}

   {{ returns.type }}
       {{ returns.desc }}
   {% endif %}
   {% endblock %}

   {% block raises %}
   {% if raises %}
   .. rubric:: {{ _('Raises') }}
         
   {% for raise in raises %}
   {{ raise.type }}
      {{ raise.desc }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block seealso %}
   {% if seealso %}
   .. rubric:: {{ _('See Also') }}
         
   {% for item in seealso %}
   :func:`{{ item }}`
      {% endfor %}
   {% endif %}
   {% endblock %}

   {% block notes %}
   {% if notes %}
   .. rubric:: {{ _('Notes') }}
   
   {% for note in notes %}
      {{ note }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block examples %}
   {% if examples %}
   .. rubric:: {{ _('Examples') }}
      
   {% for example in examples %}
      {{ example }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block references %}
   {% if references %}
   .. rubric:: {{ _('References') }}
         
   {% for reference in references %}
      {{ reference }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block todo %}
   {% if todo %}
   .. rubric:: {{ _('Todo') }}
         
   {% for item in todo %}
      {{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block warning %}
   {% if warning %}
   .. rubric:: {{ _('Warning') }}
         
   {% for item in warning %}
      {{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}