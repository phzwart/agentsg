"""Machine-checked proofs and recomputed numbers of P. H. Zwart, *The Selling closure*
(Acta Cryst. A).  Every check is a pytest test registered with ``@claim``; see
``python -m sellingproofs ledger``."""
from .registry import Claim, claim, load_all  # noqa: F401

__version__ = "1.0.0"
