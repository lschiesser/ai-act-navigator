from typing import Any

def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, set):
        return list(obj)
    return str(obj)