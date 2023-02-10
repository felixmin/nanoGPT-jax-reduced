import contextlib
import functools
from typing import Any, List
import jax
from flax.core import freeze
import numpy as np

IS_RUNNING = False

def print_compiling(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        global IS_RUNNING
        revert = False
        try:
            if not IS_RUNNING:
                print(f"compiling '{f.__name__}'")
                IS_RUNNING = True
                revert = True
            return f(*args, **kwargs)
        finally:
            if revert:
                IS_RUNNING = False
    return wrapper

LOGGER = {}
NAME = ""
LOGGER_ACTIVE = False
def log(framework: str, name: str, value: Any):
    if not LOGGER_ACTIVE:
        return
    if NAME:
        name = f"{NAME}.{name}"
    if hasattr(value, 'numpy'):
        value = value.detach().numpy()
    LOGGER.setdefault(name, {})[framework] = value

def get_logs(name: str, f=None):
    values = LOGGER[name]
    if f is not None:
        values = jax.tree_map(f, values)
    return freeze(values)

def all_close(name: str, **kwargs) -> bool:
    values = list(get_logs(name).values())
    return all(np.allclose(a, b, **kwargs) for a, b in zip(values[:-1], values[1:]))

def all_different(**kwargs) -> List[str]:
    return [name for name in LOGGER.keys() if not all_close(name, **kwargs)]

# create a context manager to append to the name
@contextlib.contextmanager
def append_name(name: str):
    global NAME
    old_name = NAME
    NAME = f"{NAME}.{name}" if NAME else name
    try:
        yield
    finally:
        NAME = old_name

@contextlib.contextmanager
def activate_logger():
    global LOGGER_ACTIVE
    old_active = LOGGER_ACTIVE
    LOGGER_ACTIVE = True
    try:
        yield
    finally:
        LOGGER_ACTIVE = old_active