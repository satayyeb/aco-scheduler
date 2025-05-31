import enum
from enum import IntEnum


class Layer(IntEnum):
    """Represents different node layers that are available in the system."""

    CLOUD = 1
    FOG = 2
    USER = 3


class FogType(IntEnum):
    """Represents different types of fog nodes that are available in the system."""

    FIXED = 1
    MOBILE = 2


class VehicleApplicationType(enum.Enum):
    CRUCIAL = 'crucial'
    HIGH_CRITICAL = 'high_critical'
    LOW_CRITICAL = 'low_critical'
