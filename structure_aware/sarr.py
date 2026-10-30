"""Compatibility imports for the former SARR API; use structure_aware.sarp."""
from .sarp import (Adapter, Graph, Node, Operation, Prepared, Result, SARP, Session,
                   output_digest)

SARR = SARP

__all__ = ["Adapter", "Graph", "Node", "Operation", "Prepared", "Result", "SARP",
           "SARR", "Session", "output_digest"]
