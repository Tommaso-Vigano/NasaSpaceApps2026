from enum import Enum


class Datatype(Enum):
    TEMPERATURE = "Temperature"
    DUST = "Dust"
    ELEVATION = "Elevation"
    IMAGERY = "Imagery"
    ALBEDO = "Albedo"
    RADIATION = "Radiation"
    THERMAL_INERTIA = "ThermalInertia"


class TemperatureUnit(Enum):
    CELSIUS = "Celsius"
    KELVIN = "Kelvin"


class RadiationUnit(Enum):
    pass


class SpatialUnit(Enum):
    Metre = "Metre"
    Coordinate = "Coordinate"
