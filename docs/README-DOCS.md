Docs make use of [Sphinx](https://www.sphinx-doc.org/en/master/) and the [furo](https://pradyunsg.me/furo/quickstart/) template.

Install the dependencies with:

```bash
pip install -r docs/requirements.txt
```

To build the docs, run:

```bash
cd docs; make html
```

**Note**: The building process produces a lot of warnings (_WARNING: duplicate label other useful arguments_). They are expected and can be ignored.
