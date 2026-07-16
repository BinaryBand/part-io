"""Self-registering CLI commands.

Command modules under this package are auto-discovered by
:func:`partio.cli.registry.discover` — importing this package is not
required; ``discover()`` walks it with ``pkgutil.walk_packages``.
"""
