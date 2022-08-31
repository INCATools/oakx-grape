from dataclasses import dataclass

GRAPE_DATA_MOD = "grape.datasets.kgobo"


def import_grape_class(name: str) -> object:
    """Dynamically import a Grape class based on its reference.

    :param reference: The reference or path for the class to be imported.
    :return: The imported class
    """
    mod = __import__(GRAPE_DATA_MOD, fromlist=[name])
    this_class = getattr(mod, name)
    return this_class


@dataclass
class GrapeImplementation(BaseOntologyInterface):

    def __post_init__(self):
        pass

