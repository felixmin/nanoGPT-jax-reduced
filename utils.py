import functools

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
