"""Ports: the Protocol interfaces that adapters must satisfy.

A port is defined here in core; its concrete implementation lives in adapters
and is wired to the application in app. Because core may not import adapters,
the dependency always points inward. `ty` verifies each adapter structurally
satisfies its port at the point where app wires them together.
"""
