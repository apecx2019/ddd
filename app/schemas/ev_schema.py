from pydantic import BaseModel

class EVInput(BaseModel):
    year: int
    month: int
    season: int
    charging_station: int
    population_density: float
    gdp_per_capita: float
    average_income: float