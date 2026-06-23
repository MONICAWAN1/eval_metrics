import functools as ft
import traceback as tb


class ExceptionPrinter:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        except Exception as e:
            tb.print_exception(e)
            raise

    def __getattr__(self, attr):
        # Called during unpickling (e.g. submitit) before internal state is restored; report
        # any attribute as not found to avoid infinite recursion.
        if "f" not in self.__dict__:
            raise AttributeError()
        # Hack so hydra.main can read f.__code__ to determine the calling file.
        return getattr(self.f, attr)


def print_exceptions(f):
    """Print any exception raised by the annotated function to stderr.

    Helpful when an outer function swallows exceptions, such as hydra's submitit launcher.
    """
    return ft.wraps(f)(ExceptionPrinter(f))
