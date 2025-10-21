__version__ = "0.1.0"
__author__ = "Wilker Aziz"
__license__ = "MIT"

# Optional: utility to load module dynamically
def load(module_name: str):
    """
    Dynamically load a submodule, e.g.:
        m3 = pgmini.load("m3")
    """
    import importlib
    return importlib.import_module(f"pgmini.{module_name}")

