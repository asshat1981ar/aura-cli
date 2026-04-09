from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, ValidationError

class RouteRequest(BaseModel):
    source: str
    destination: str

class Router:
    def __init__(self, routes: List[RouteRequest]) -> None:
        self.routes = routes

    def find_route(self, source: str, destination: str) -> Optional[RouteRequest]:
        for route in self.routes:
            if route.source == source and route.destination == destination:
                return route
        return None

    def add_route(self, route: RouteRequest) -> None:
        if self.find_route(route.source, route.destination) is not None:
            raise ValueError('Route already exists.')
        self.routes.append(route)

    def validate_routes(self) -> None:
        for route in self.routes:
            try:
                RouteRequest(**route.dict())
            except ValidationError as e:
                print(f'Invalid route: {e}')