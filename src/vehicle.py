"""Veículo e direções.

O veículo é um modelo 1D: ``position`` é a distância (em pixels) da FRENTE do
veículo até o ponto de origem, medida ao longo da sua faixa. A conversão para
coordenadas de tela é feita pela simulação (``Geometry``), mantendo esta classe
independente de qualquer elemento gráfico.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .traffic_light import Group


class Direction(Enum):
    """Direção de ORIGEM do veículo (de onde ele vem)."""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

    @property
    def opposite(self) -> "Direction":
        return _OPPOSITE[self]

    @property
    def group(self) -> Group:
        return Group.NORTH_SOUTH if self in (Direction.NORTH, Direction.SOUTH) else Group.EAST_WEST

    @property
    def label(self) -> str:
        return _LABELS[self]


_OPPOSITE = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}
_LABELS = {
    Direction.NORTH: "Norte",
    Direction.SOUTH: "Sul",
    Direction.EAST: "Leste",
    Direction.WEST: "Oeste",
}


class SignalDecision(Enum):
    """Decisão do motorista diante de um sinal não-verde (tomada uma vez)."""

    STOP = "STOP"
    GO = "GO"


@dataclass
class Vehicle:
    id: int
    origin: Direction
    destination: Direction
    position: float            # px, frente do veículo ao longo da faixa
    speed: float               # px/s
    max_speed: float           # px/s
    acceleration: float        # px/s²
    deceleration: float        # px/s² (positivo)
    length: float = 24.0
    width: float = 14.0
    color: Tuple[int, int, int] = (200, 200, 200)
    stopped_speed_threshold: float = 5.0
    spawn_time: float = 0.0

    # estado dinâmico
    waiting_time: float = 0.0          # s acumulados parado antes da linha de retenção
    distance_traveled: float = 0.0     # px percorridos desde o nascimento
    is_braking: bool = False
    passed_stop_line: bool = False     # frente já passou da linha de retenção
    has_crossed: bool = False          # já liberou completamente a interseção
    collided: bool = False             # já esteve envolvido em alguma colisão
    signal_decision: Optional[SignalDecision] = None
    yellow_runner: bool = False

    @property
    def is_stopped(self) -> bool:
        return self.speed < self.stopped_speed_threshold

    @property
    def is_moving(self) -> bool:
        return not self.is_stopped

    def update(self, dt: float, target_speed: float) -> None:
        """Acelera/freia em direção a ``target_speed`` e avança a posição.

        A velocidade é atualizada antes da posição (integração semi-implícita),
        o que é levemente conservador ao frear.
        """
        if self.speed < target_speed:
            self.speed = min(target_speed, self.speed + self.acceleration * dt)
            self.is_braking = False
        elif self.speed > target_speed:
            self.speed = max(target_speed, self.speed - self.deceleration * dt)
            self.is_braking = True
        else:
            self.is_braking = False

        # Evita "rastejar" indefinidamente quando o alvo é parar.
        if target_speed <= 0.0 and self.speed < self.stopped_speed_threshold * 0.6:
            self.speed = 0.0

        step = self.speed * dt
        self.position += step
        self.distance_traveled += step
