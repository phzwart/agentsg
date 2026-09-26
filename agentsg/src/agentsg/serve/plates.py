"""ITA plate rendering (optional matplotlib)."""
from __future__ import annotations

from io import BytesIO
from threading import Lock

_CACHE: dict[tuple, bytes] = {}
_LOCK = Lock()
_MAX_CACHE = 64


def plates_available() -> bool:
    """True if matplotlib can be imported."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def render_ita_png(sg, *, projection: str, legend: bool, show_centring: bool) -> bytes:
    """Render ``ita_plate`` to PNG bytes. Raises ImportError if no matplotlib."""
    key = (str(sg), projection, bool(legend), bool(show_centring))
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from ..cell.diagrams import ita_plate

    fig = ita_plate(sg, legend=legend, show_centring=show_centring,
                    projection=projection)
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    finally:
        plt.close(fig)
    data = buf.getvalue()
    with _LOCK:
        if len(_CACHE) >= _MAX_CACHE:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = data
    return data
