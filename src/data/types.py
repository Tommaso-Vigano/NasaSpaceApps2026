from enum import Enum


class Datatype(Enum):
    TEMPERATURE = "Temperature"
    DUST = "Dust"
    ELEVATION = "Elevation"
    ALBEDO = "Albedo"
    RADIATION = "Radiation"
    PENALTY = "Penalty"
    THERMAL_INERTIA = "Thermal Inertia"



class TemperatureUnit(Enum):
    CELSIUS = "Celsius"
    KELVIN = "Kelvin"


class RadiationUnit(Enum):
    pass


class SpatialUnit(Enum):
    Metre = "Metre"
    Coordinate = "Coordinate"
