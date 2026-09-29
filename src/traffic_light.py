"""Semáforo (máquina de estados segura) e controladores de fase.

Separação de responsabilidades:

* ``TrafficLight``  -> garante a SEGURANÇA. Nenhum controlador (nem a futura IA)
  consegue colocar os dois grupos em verde ao mesmo tempo, pois só existe a
  operação ``request_change()``.
* ``TrafficController`` -> decide QUANDO pedir a troca de fase. Hoje existe
  apenas o ``FixedTimeController`` (tempo fixo). No futuro, uma rede neural
  poderá implementar a mesma interface ``decide(sensor_data) -> Decision``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # apenas para type hints (evita import circular)
    from .simulation import SensorData


class LightState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class Group(Enum):
    """Grupos de semáforos compatíveis entre si."""

    NORTH_SOUTH = "NORTH_SOUTH"
    EAST_WEST = "EAST_WEST"


class Phase(Enum):
    """Fases do ciclo completo. A ordem do ciclo está em ``_NEXT_PHASE``."""

    NS_GREEN = "NS_GREEN"
    NS_YELLOW = "NS_YELLOW"
    ALL_RED_TO_EW = "ALL_RED_TO_EW"  # vermelho total antes de liberar Leste/Oeste
    EW_GREEN = "EW_GREEN"
    EW_YELLOW = "EW_YELLOW"
    ALL_RED_TO_NS = "ALL_RED_TO_NS"  # vermelho total antes de liberar Norte/Sul


class Decision(Enum):
    """Saída de qualquer controlador (fixo hoje, rede neural amanhã)."""

    KEEP = "KEEP"
    CHANGE_PHASE = "CHANGE_PHASE"


@dataclass
class TrafficLightConfig:
    min_green_time: float = 8.0   # s - verde mínimo antes de aceitar troca
    yellow_time: float = 3.0      # s - duração do amarelo
    min_red_time: float = 1.5     # s - vermelho total (limpeza da interseção)


# (estado do grupo Norte/Sul, estado do grupo Leste/Oeste) em cada fase
_PHASE_STATES = {
    Phase.NS_GREEN: (LightState.GREEN, LightState.RED),
    Phase.NS_YELLOW: (LightState.YELLOW, LightState.RED),
    Phase.ALL_RED_TO_EW: (LightState.RED, LightState.RED),
    Phase.EW_GREEN: (LightState.RED, LightState.GREEN),
    Phase.EW_YELLOW: (LightState.RED, LightState.YELLOW),
    Phase.ALL_RED_TO_NS: (LightState.RED, LightState.RED),
}

_NEXT_PHASE = {
    Phase.NS_GREEN: Phase.NS_YELLOW,
    Phase.NS_YELLOW: Phase.ALL_RED_TO_EW,
    Phase.ALL_RED_TO_EW: Phase.EW_GREEN,
    Phase.EW_GREEN: Phase.EW_YELLOW,
    Phase.EW_YELLOW: Phase.ALL_RED_TO_NS,
    Phase.ALL_RED_TO_NS: Phase.NS_GREEN,
}

_GREEN_PHASES = (Phase.NS_GREEN, Phase.EW_GREEN)
_YELLOW_PHASES = (Phase.NS_YELLOW, Phase.EW_YELLOW)
_ALL_RED_PHASES = (Phase.ALL_RED_TO_EW, Phase.ALL_RED_TO_NS)

# Verificação estática da tabela: em nenhuma fase os dois grupos podem estar
# liberados (verde/amarelo) ao mesmo tempo.
for _ns, _ew in _PHASE_STATES.values():
    assert _ns is LightState.RED or _ew is LightState.RED, "fase insegura na tabela"


class TrafficLight:
    """Semáforo de interseção de quatro direções (dois grupos)."""

    def __init__(self, config: TrafficLightConfig | None = None,
                 initial_phase: Phase = Phase.NS_GREEN):
        self.config = config or TrafficLightConfig()
        self.phase = initial_phase
        self.time_in_phase = 0.0
        self.phase_changes = 0

    # ------------------------------------------------------------------ consulta
    def state_for(self, group: Group) -> LightState:
        ns, ew = _PHASE_STATES[self.phase]
        return ns if group is Group.NORTH_SOUTH else ew

    @property
    def is_green_phase(self) -> bool:
        return self.phase in _GREEN_PHASES

    @property
    def can_change(self) -> bool:
        """True se um pedido de troca seria aceito agora."""
        return self.is_green_phase and self.time_in_phase >= self.config.min_green_time

    # ------------------------------------------------------------------ comandos
    def request_change(self) -> bool:
        """Solicita o fim do verde atual. É a ÚNICA porta de entrada para IA.

        Retorna True se aceito. Se o verde mínimo ainda não passou (ou se já
        estamos em amarelo/vermelho total), o pedido é ignorado.
        """
        if not self.can_change:
            return False
        self._advance()
        self.time_in_phase = 0.0
        return True

    def update(self, dt: float) -> None:
        """Avança o relógio. Amarelo e vermelho total terminam sozinhos."""
        self.time_in_phase += dt
        if self.phase in _YELLOW_PHASES:
            duration = self.config.yellow_time
        elif self.phase in _ALL_RED_PHASES:
            duration = self.config.min_red_time
        else:
            return  # verde só termina via request_change()
        if self.time_in_phase >= duration:
            self.time_in_phase -= duration
            self._advance()

    def _advance(self) -> None:
        self.phase = _NEXT_PHASE[self.phase]
        self.phase_changes += 1


# ---------------------------------------------------------------------- controllers
class TrafficController:
    """Interface: recebe dados de sensores e devolve uma ``Decision``."""

    def decide(self, data: "SensorData") -> Decision:
        raise NotImplementedError

    def reset(self) -> None:
        """Chamado quando a simulação reinicia."""


class FixedTimeController(TrafficController):
    """Controlador de tempo fixo: pede troca após ``green_time`` segundos."""

    def __init__(self, green_time: float = 12.0):
        self.green_time = green_time

    def decide(self, data: "SensorData") -> Decision:
        if data.can_change_phase and data.time_since_phase_change >= self.green_time:
            return Decision.CHANGE_PHASE
        return Decision.KEEP
