# Constants for ERA5 dataset

# 37 pressure levels available from graphcast stats
pressure_levels = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]


# ArchesWeather default settings for ERA5 dataset.
arches_default_pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

arches_default_level_variables = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "vertical_velocity",
]

arches_default_surface_variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]


# Short names for variables used in tensordicts and metrics
surface_variables_short = {
    "10m_u_component_of_wind": "U10m",
    "10m_v_component_of_wind": "V10m",
    "2m_temperature": "T2m",
    "mean_sea_level_pressure": "MSLP",
    "low_vegetation_cover": "CVL",
    "high_vegetation_cover": "CVH",
    "tympe_of_low_vegetation_cover": "TVL",
    "type_of_high_vegetation_cover": "TVH",
    "soil_type": "SLT",
    "standard_deviation_of_filtred_subgrid_orography": "SDFSOR",
    "angle_of_sub_gridscale_orography": "ANOR",
    "anisotropy_of_subgridscale_orography": "ASOR",
    "geopotential_at_surface": "Z0",
    "lake_cover": "LC",
    "lake_depth": "LD",
    "sea_ice_cover": "SIC",
    "sea_surface_temperature": "SST",
    "slope_of_subgridscale_orography": "SSOR",
    "standard_deviation_of_orography": "SDFO",
    "surface_pressure": "SP",
    "toa_incident_solar_radiation": "SIS",
    "toa_incident_solar_radiation_12hr": "SIS12",
    "toa_incident_solar_radiation_24hr": "SIS24",
    "total_cloud_cover": "TCC",
    "total_precipitation_12hr": "TP",
    "total_precipitation_24hr": "TP24",
    "total_column_water_vapour": "TCWV",
    "wind_speed": "WS",
}

level_variables_short = {
    "geopotential": "Z",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "temperature": "T",
    "specific_humidity": "Q",
    "vertical_velocity": "W",
}
