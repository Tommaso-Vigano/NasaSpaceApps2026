from enum import Enum


class Datatype(Enum):
    TEMPERATURE = "Temperature"
    DUST = "Dust"
    ELEVATION = "Elevation"
    ALBEDO = "Albedo"
    RADIATION = "Radiation"
    PENALTY = "Penalty"
    THERMAL_INERTIA = "Thermal Inertia"
    TERRAIN_ETA = "Terrain Eta"
    STEP_HEIGHT = "Step Height"            # m, gradino massimo |step| della cella
    LANDER_DISTANCE = "Lander Distance"    # m
    RETURN_TIME = "Return Time"            # s, cammino minimo dalla cella al lander
    EVA_TIME = "EVA Time"                  # s, andata + ritorno passando per la cella
    VISIBILITY = "Visibility"              # 0 = in vista del lander, 1 = fuori vista
    DUST_TAU = "Dust Optical Depth"        # uniforme (MCD)
    SOLAR_FLUX = "Solar Flux"              # W/m2, uniforme (MCD)
    WIND_SPEED = "Wind Speed"              # m/s, uniforme (MCD)



class TemperatureUnit(Enum):
    CELSIUS = "Celsius"
    KELVIN = "Kelvin"


class RadiationUnit(Enum):
    pass


class SpatialUnit(Enum):
    Metre = "Metre"
    Coordinate = "Coordinate"
