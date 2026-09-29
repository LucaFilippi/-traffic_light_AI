"""Núcleo do simulador Traffic Lights AI (pré-alpha v0.1).

Organização deste arquivo:

1. ``SimulationConfig`` ..... todas as configurações centralizadas
2. ``Geometry`` ............. conversão de "distância na faixa" -> pixels
3. ``SensorData`` / ``Metrics``  dados expostos para a futura IA / avaliação
4. ``Simulation`` ........... lógica (``step``) SEPARADA da renderização (``render``)

A lógica (``step``, ``get_sensor_data``...) não importa nem usa Pygame. O Pygame
só é importado dentro de ``run()``, então a simulação pode rodar "headless"
(``run_headless``) para treinamento rápido no futuro.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .traffic_light import (
    Decision,
    FixedTimeController,
    LightState,
    Phase,
    TrafficController,
    TrafficLight,
    TrafficLightConfig,
)
from .vehicle import Direction, SignalDecision, Vehicle

Color = Tuple[int, int, int]


# ============================================================================
# 1. CONFIGURAÇÃO
# ============================================================================
@dataclass
class SimulationConfig:
    # --- janela / loop -------------------------------------------------------
    window_width: int = 1000
    window_height: int = 700
    fps: int = 60
    window_title: str = "Traffic Lights AI - Pre-Alpha v0.1"

    # --- tempo ---------------------------------------------------------------
    physics_dt: float = 1.0 / 60.0    # passo fixo da simulação (s)
    max_frame_time: float = 0.1       # limita frames muito longos (s)
    time_scale: float = 1.0           # velocidade da simulação (1x, 2x, ...)
    decision_interval: float = 0.25   # de quanto em quanto tempo o controlador decide (s)

    # --- geometria -----------------------------------------------------------
    lane_width: float = 30.0          # largura de UMA faixa (px). Rua = 2 faixas.

    # --- tráfego -------------------------------------------------------------
    seed: Optional[int] = 42          # None = seed aleatória (exibida na tela)
    traffic_density: float = 0.5      # 1.0 = um carro a cada ~spawn_interval_base s por direção
    direction_bias: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    spawn_interval_base: float = 3.0  # s (com densidade 1.0)
    spawn_interval_jitter: float = 0.5   # variação relativa do intervalo (+-50%)
    spawn_position_jitter: float = 25.0  # px de variação da posição inicial
    spawn_initial_speed_factor: float = 0.6  # fração da vel. máx. ao nascer
    spawn_retry_interval: float = 0.25   # s até tentar de novo se não houver espaço
    speed_variation: float = 0.15    # ±15% de variação na velocidade ideal de spawn

    # --- veículos (faixas sorteadas por veículo) -----------------------------
    vehicle_length: float = 24.0
    vehicle_width: float = 14.0
    max_speed_range: Tuple[float, float] = (80.0, 120.0)        # px/s
    acceleration_range: Tuple[float, float] = (60.0, 90.0)      # px/s²
    deceleration_range: Tuple[float, float] = (140.0, 200.0)    # px/s²
    vehicle_colors: Tuple[Color, ...] = (
        (231, 76, 60), (52, 152, 219), (241, 196, 15), (155, 89, 182),
        (26, 188, 156), (230, 126, 34), (236, 240, 241), (52, 73, 94),
    )

    # --- comportamento -------------------------------------------------------
    min_gap: float = 8.0              # px - distância mínima parado atrás do líder
    time_headway: float = 0.6         # s - folga extra proporcional à velocidade
    headway_gain: float = 1.5         # 1/s - quão rápido reabre espaço se estiver colado
    braking_comfort: float = 0.85     # fração da desaceleração usada no planejamento
    stop_line_margin: float = 3.0     # px - quanto antes da linha o carro para
    stopped_speed_threshold: float = 5.0  # px/s - abaixo disso = "parado"

    # --- sensores ------------------------------------------------------------
    sensor_range: float = 250.0       # px antes da linha de retenção

    # --- semáforo ------------------------------------------------------------
    light: TrafficLightConfig = field(default_factory=TrafficLightConfig)
    fixed_green_time: float = 12.0    # s - controlador de tempo fixo
    yellow_run_probability: float = 0.08  # chance de um motorista "arriscar" o amarelo
    yellow_run_boost: float = 0.35        # aceleração extra quando ele decide cruzar


# ============================================================================
# 2. GEOMETRIA
# ============================================================================
class Geometry:
    """Interseção de 4 direções, mão direita (Brasil), 1 faixa por sentido.

    Cada veículo tem uma coordenada ``s`` (distância da frente até o ponto de
    origem, na borda da janela). Aqui convertemos ``s`` em (x, y) de tela.
    """

    def __init__(self, cfg: SimulationConfig):
        self.width = cfg.window_width
        self.height = cfg.window_height
        self.cx = self.width / 2
        self.cy = self.height / 2
        self.lane_width = cfg.lane_width
        self.half = cfg.lane_width          # meia largura da interseção
        self.box_size = 2 * self.half

    @staticmethod
    def is_vertical(d: Direction) -> bool:
        return d in (Direction.NORTH, Direction.SOUTH)

    def stop_s(self, d: Direction) -> float:
        """Posição da linha de retenção (início da interseção)."""
        return (self.cy if self.is_vertical(d) else self.cx) - self.half

    def box_end_s(self, d: Direction) -> float:
        return self.stop_s(d) + self.box_size

    def total_length(self, d: Direction) -> float:
        return self.height if self.is_vertical(d) else self.width

    def heading(self, d: Direction) -> Tuple[float, float]:
        """Vetor unitário do movimento na tela (y cresce para baixo)."""
        return {
            Direction.NORTH: (0.0, 1.0),    # vem do norte, vai para baixo
            Direction.SOUTH: (0.0, -1.0),
            Direction.EAST: (-1.0, 0.0),    # vem do leste, vai para a esquerda
            Direction.WEST: (1.0, 0.0),
        }[d]

    def world_pos(self, d: Direction, s: float) -> Tuple[float, float]:
        lane = self.lane_width / 2
        if d is Direction.NORTH:
            return self.cx - lane, s
        if d is Direction.SOUTH:
            return self.cx + lane, self.height - s
        if d is Direction.EAST:
            return self.width - s, self.cy - lane
        return s, self.cy + lane  # WEST

    def light_pos(self, d: Direction) -> Tuple[float, float]:
        """Onde desenhar o semáforo de cada aproximação."""
        off = self.half + 14
        return {
            Direction.NORTH: (self.cx - off, self.cy - off),
            Direction.SOUTH: (self.cx + off, self.cy + off),
            Direction.EAST: (self.cx + off, self.cy - off),
            Direction.WEST: (self.cx - off, self.cy + off),
        }[d]

    def stop_line_rect(self, d: Direction) -> Tuple[float, float, float, float]:
        h, lw, t = self.half, self.lane_width, 3
        return {
            Direction.NORTH: (self.cx - lw, self.cy - h - t, lw, t),
            Direction.SOUTH: (self.cx, self.cy + h, lw, t),
            Direction.EAST: (self.cx + h, self.cy - lw, t, lw),
            Direction.WEST: (self.cx - h - t, self.cy, t, lw),
        }[d]

    def vehicle_rect(self, v: Vehicle) -> Tuple[float, float, float, float]:
        """Retângulo alinhado aos eixos (x0, y0, x1, y1) do veículo."""
        fx, fy = self.world_pos(v.origin, v.position)
        rx, ry = self.world_pos(v.origin, v.position - v.length)
        hw = v.width / 2
        if self.is_vertical(v.origin):
            return fx - hw, min(fy, ry), fx + hw, max(fy, ry)
        return min(fx, rx), fy - hw, max(fx, rx), fy + hw


# ============================================================================
# 3. DADOS EXPOSTOS (sensores e métricas)
# ============================================================================
@dataclass
class SensorData:
    """Fotografia do estado, sem expor objetos internos da simulação.

    Esta é a ÚNICA coisa que a futura rede neural precisará enxergar.
    """

    nearby_vehicles: Dict[Direction, int]    # veículos na zona do sensor (antes da linha)
    waiting_vehicles: Dict[Direction, int]   # desses, quantos estão parados
    average_waiting_time: float              # s, média dos veículos parados
    average_speed: float                     # px/s, média dos veículos ativos
    phase: Phase
    time_since_phase_change: float           # s
    can_change_phase: bool                   # um pedido de troca seria aceito agora?

    def to_vector(self) -> List[float]:
        """Vetor numérico BRUTO (sem normalização) para uso futuro na rede."""
        order = list(Direction)
        return (
            [float(self.nearby_vehicles[d]) for d in order]
            + [float(self.waiting_vehicles[d]) for d in order]
            + [
                self.average_waiting_time,
                self.average_speed,
                float(list(Phase).index(self.phase)),
                self.time_since_phase_change,
            ]
        )


@dataclass
class Metrics:
    """Contadores acumulados; matéria-prima do futuro sistema de fitness."""

    vehicles_spawned: int = 0
    vehicles_completed: int = 0
    collisions: int = 0
    completed_wait_time: float = 0.0   # soma do tempo de espera dos que já saíram


# ============================================================================
# 4. SIMULAÇÃO
# ============================================================================
class Simulation:
    def __init__(self, config: Optional[SimulationConfig] = None,
                 controller: Optional[TrafficController] = None):
        self.config = config or SimulationConfig()
        self.geometry = Geometry(self.config)
        self.controller: TrafficController = controller or FixedTimeController(
            self.config.fixed_green_time
        )
        self.seed: int = (
            self.config.seed if self.config.seed is not None else random.randrange(2**31)
        )
        self.paused = False
        self.time_scale = self.config.time_scale
        self.show_sensors = False
        self.reset()

    # ------------------------------------------------------------------ estado
    def reset(self, seed: Optional[int] = None) -> None:
        """Reinicia tudo. Com a mesma seed, a mesma situação se repete."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.time = 0.0
        self.light = TrafficLight(self.config.light)
        self.controller.reset()
        self.lanes: Dict[Direction, List[Vehicle]] = {d: [] for d in Direction}  # líder primeiro
        self.metrics = Metrics()
        self._next_id = 0
        self._next_spawn: Dict[Direction, float] = {
            d: self.rng.uniform(0.0, 2.0) for d in Direction
        }
        self._decision_timer = 0.0
        self._active_collisions: Set[Tuple[int, int]] = set()

    # ------------------------------------------------------------------ consultas
    @property
    def vehicles(self) -> List[Vehicle]:
        return [v for lane in self.lanes.values() for v in lane]

    @property
    def vehicle_count(self) -> int:
        return sum(len(lane) for lane in self.lanes.values())

    @property
    def waiting_count(self) -> int:
        """Veículos parados antes da linha de retenção (em qualquer distância)."""
        return sum(
            1 for v in self.vehicles
            if v.position >= 0 and not v.passed_stop_line and v.is_stopped
        )

    def get_sensor_data(self) -> SensorData:
        cfg, geo = self.config, self.geometry
        nearby: Dict[Direction, int] = {}
        waiting: Dict[Direction, int] = {}
        wait_times: List[float] = []
        speeds: List[float] = []
        for d, lane in self.lanes.items():
            stop_s = geo.stop_s(d)
            n = w = 0
            for v in lane:
                if v.position >= 0:
                    speeds.append(v.speed)
                if stop_s - cfg.sensor_range <= v.position < stop_s:
                    n += 1
                    if v.is_stopped:
                        w += 1
                        wait_times.append(v.waiting_time)
            nearby[d], waiting[d] = n, w
        return SensorData(
            nearby_vehicles=nearby,
            waiting_vehicles=waiting,
            average_waiting_time=sum(wait_times) / len(wait_times) if wait_times else 0.0,
            average_speed=sum(speeds) / len(speeds) if speeds else 0.0,
            phase=self.light.phase,
            time_since_phase_change=self.light.time_in_phase,
            can_change_phase=self.light.can_change,
        )

    def summary(self) -> str:
        m = self.metrics
        avg_wait = m.completed_wait_time / m.vehicles_completed if m.vehicles_completed else 0.0
        return (
            f"t={self.time:.1f}s seed={self.seed} spawned={m.vehicles_spawned} "
            f"completed={m.vehicles_completed} active={self.vehicle_count} "
            f"collisions={m.collisions} avg_wait_completed={avg_wait:.2f}s "
            f"phase_changes={self.light.phase_changes}"
        )

    # ------------------------------------------------------------------ LÓGICA
    def step(self, dt: float) -> None:
        """Avança a simulação em ``dt`` segundos. Sem qualquer código gráfico."""
        self.time += dt

        # 1) controlador decide (em intervalos) e o semáforo valida o pedido
        self._decision_timer += dt
        if self._decision_timer >= self.config.decision_interval:
            self._decision_timer -= self.config.decision_interval
            if self.controller.decide(self.get_sensor_data()) is Decision.CHANGE_PHASE:
                self.light.request_change()

        # 2) semáforo, geração e movimento
        self.light.update(dt)
        self._spawn_vehicles()
        self._update_vehicles(dt)

        # 3) colisões
        self._detect_collisions()

    def run_headless(self, duration: float, dt: Optional[float] = None) -> None:
        """Roda sem janela nem renderização (base para treino futuro)."""
        dt = dt or self.config.physics_dt
        end = self.time + duration
        while self.time < end:
            self.step(dt)

    # ---- geração de tráfego
    def _spawn_interval_for(self, d: Direction, density: float) -> float:
        cfg = self.config
        bias = {
            Direction.NORTH: cfg.direction_bias[0],
            Direction.SOUTH: cfg.direction_bias[1],
            Direction.EAST: cfg.direction_bias[2],
            Direction.WEST: cfg.direction_bias[3],
        }[d]
        jitter = cfg.spawn_interval_jitter
        base = cfg.spawn_interval_base / max(density, 0.01)
        return (base / max(bias, 0.05)) * self.rng.uniform(1.0 - jitter, 1.0 + jitter)

    def _spawn_vehicles(self) -> None:
        for d in self.rng.sample(list(Direction), k=len(Direction)):
            if self.time < self._next_spawn[d]:
                continue
            if self._try_spawn(d):
                self._next_spawn[d] = self.time + self._spawn_interval_for(d, self.config.traffic_density)
            else:
                self._next_spawn[d] = self.time + self.config.spawn_retry_interval

    def _try_spawn(self, d: Direction) -> bool:
        cfg, rng = self.config, self.rng
        # Sorteios sempre na mesma ordem -> reprodutível com a mesma seed.
        max_speed = rng.uniform(*cfg.max_speed_range)
        max_speed *= rng.uniform(1.0 - cfg.speed_variation, 1.0 + cfg.speed_variation)
        accel = rng.uniform(*cfg.acceleration_range)
        decel = rng.uniform(*cfg.deceleration_range)
        position = -rng.uniform(0.0, cfg.spawn_position_jitter)
        color = cfg.vehicle_colors[rng.randrange(len(cfg.vehicle_colors))]
        speed = max_speed * cfg.spawn_initial_speed_factor * rng.uniform(0.8, 1.15)
        yellow_runner = rng.random() < cfg.yellow_run_probability

        lane = self.lanes[d]
        if lane:
            last = lane[-1]
            gap = last.position - last.length - position - cfg.min_gap
            if gap <= 0:
                return False  # sem espaço; tenta de novo em instantes
            b = decel * cfg.braking_comfort
            safe = math.sqrt(2 * b * (gap + last.speed ** 2 / (2 * last.deceleration)))
            speed = min(speed, safe)

        self.lanes[d].append(Vehicle(
            id=self._next_id,
            origin=d,
            destination=d.opposite,
            position=position,
            speed=speed,
            max_speed=max_speed,
            acceleration=accel,
            deceleration=decel,
            length=cfg.vehicle_length,
            width=cfg.vehicle_width,
            color=color,
            stopped_speed_threshold=cfg.stopped_speed_threshold,
            spawn_time=self.time,
            yellow_runner=yellow_runner,
        ))
        self._next_id += 1
        self.metrics.vehicles_spawned += 1
        return True

    # ---- movimento
    def _update_vehicles(self, dt: float) -> None:
        geo = self.geometry
        for d, lane in self.lanes.items():
            stop_s = geo.stop_s(d)
            box_end = geo.box_end_s(d)
            light_state = self.light.state_for(d.group)
            leader: Optional[Vehicle] = None
            for v in lane:
                v.update(dt, self._target_speed(v, leader, stop_s, light_state))
                v.passed_stop_line = v.position >= stop_s
                v.has_crossed = (v.position - v.length) >= box_end
                if v.position >= 0 and not v.passed_stop_line and v.is_stopped:
                    v.waiting_time += dt
                leader = v

            # remove quem saiu da área (a lista está ordenada: líder primeiro)
            total = geo.total_length(d)
            while lane and (lane[0].position - lane[0].length) >= total:
                gone = lane.pop(0)
                self.metrics.vehicles_completed += 1
                self.metrics.completed_wait_time += gone.waiting_time

    def _target_speed(self, v: Vehicle, leader: Optional[Vehicle],
                      stop_s: float, light_state: LightState) -> float:
        """Maior velocidade segura agora. O veículo tenta se aproximar dela.

        Regra base: velocidade v é segura se der para parar antes do obstáculo:
        ``v <= sqrt(2 * b * distância_livre)``.
        """
        cfg = self.config
        b = v.deceleration * cfg.braking_comfort
        target = v.max_speed

        # --- veículo à frente (assume que ele pode frear no máximo com sua decel.)
        if leader is not None:
            gap = leader.position - leader.length - v.position
            # (1) segurança física: dá para parar antes de bater, mesmo que o
            #     líder freie o máximo que consegue.
            free = max(gap - cfg.min_gap, 0.0)
            leader_stop = leader.speed ** 2 / (2 * leader.deceleration)
            target = min(target, math.sqrt(2 * b * (free + leader_stop)))
            # (2) folga proporcional à velocidade: se estiver perto demais de um
            #     líder que se move, fica mais lento que ele para reabrir espaço.
            if leader.speed > cfg.stopped_speed_threshold:
                deficit = gap - cfg.min_gap - cfg.time_headway * v.speed
                if deficit < 0:
                    target = min(target, max(leader.speed + cfg.headway_gain * deficit, 0.0))

        # --- semáforo (só vale antes da linha de retenção)
        if v.position < stop_s:
            if light_state is LightState.GREEN:
                v.signal_decision = None
            elif light_state is LightState.YELLOW:
                dist = max(stop_s - v.position - cfg.stop_line_margin, 0.0)
                if v.signal_decision is None:
                    # 8% dos motoristas "arrisca" o amarelo e tenta cruzar.
                    if v.yellow_runner and v.speed > cfg.stopped_speed_threshold:
                        v.signal_decision = SignalDecision.GO
                    else:
                        can_stop = v.speed ** 2 / (2 * b) <= dist
                        v.signal_decision = SignalDecision.STOP if can_stop else SignalDecision.GO
                if v.signal_decision is SignalDecision.STOP:
                    target = min(target, math.sqrt(2 * b * dist))
                elif leader is None:
                    target = max(target, min(v.max_speed, v.speed + v.acceleration * cfg.yellow_run_boost))
            else:
                dist = max(stop_s - v.position - cfg.stop_line_margin, 0.0)
                if v.signal_decision is None:
                    # Decisão tomada UMA vez por sinal: consegue parar? Então para.
                    can_stop = v.speed ** 2 / (2 * b) <= dist
                    v.signal_decision = SignalDecision.STOP if can_stop else SignalDecision.GO
                if v.signal_decision is SignalDecision.STOP:
                    target = min(target, math.sqrt(2 * b * dist))
        return target

    # ---- colisões
    def _detect_collisions(self) -> None:
        geo = self.geometry
        current: Set[Tuple[int, int]] = set()

        # (a) mesma faixa: batida traseira
        for lane in self.lanes.values():
            for leader, follower in zip(lane, lane[1:]):
                if follower.position > leader.position - leader.length:
                    self._register_collision(leader, follower, current)

        # (b) tráfego cruzado: só quem está dentro/na área da interseção
        inside: List[Vehicle] = []
        for d, lane in self.lanes.items():
            stop_s, box_end = geo.stop_s(d), geo.box_end_s(d)
            inside.extend(v for v in lane if v.position > stop_s and v.position - v.length < box_end)
        for i, a in enumerate(inside):
            for b in inside[i + 1:]:
                if a.origin.group is b.origin.group:
                    continue
                ax0, ay0, ax1, ay1 = geo.vehicle_rect(a)
                bx0, by0, bx1, by1 = geo.vehicle_rect(b)
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    self._register_collision(a, b, current)

        self._active_collisions = current  # colisões terminadas saem do conjunto

    def _register_collision(self, a: Vehicle, b: Vehicle, current: Set[Tuple[int, int]]) -> None:
        key = (min(a.id, b.id), max(a.id, b.id))
        current.add(key)
        if key not in self._active_collisions:
            self.metrics.collisions += 1   # conta cada par UMA vez
        a.collided = b.collided = True

    # ========================================================================
    # RENDERIZAÇÃO E LOOP PRINCIPAL (únicos trechos que usam Pygame)
    # ========================================================================
    def run(self) -> None:
        """Abre a janela e roda até o usuário fechar."""
        import pygame  # import tardio: a lógica não depende de Pygame

        self._pg = pygame
        pygame.init()
        try:
            cfg = self.config
            self._screen = pygame.display.set_mode((cfg.window_width, cfg.window_height))
            pygame.display.set_caption(cfg.window_title)
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("dejavusansmono,consolas,couriernew,monospace", 15)
            self._font_small = pygame.font.SysFont("dejavusansmono,consolas,couriernew,monospace", 12)
            self._running = True
            accumulator = 0.0

            while self._running:
                # delta time real do frame (limitado) -> passos fixos da simulação
                frame_dt = min(self._clock.tick(cfg.fps) / 1000.0, cfg.max_frame_time)
                for event in pygame.event.get():
                    self._handle_event(event)
                if not self.paused:
                    accumulator += frame_dt * self.time_scale
                    while accumulator >= cfg.physics_dt:
                        self.step(cfg.physics_dt)
                        accumulator -= cfg.physics_dt
                self.render()
                pygame.display.flip()
        finally:
            pygame.quit()

    def _handle_event(self, event) -> None:
        pg = self._pg
        if event.type == pg.QUIT:
            self._running = False
        elif event.type == pg.KEYDOWN:
            k = event.key
            if k == pg.K_ESCAPE:
                self._running = False
            elif k == pg.K_SPACE:
                self.paused = not self.paused
            elif k == pg.K_r:
                self.reset()                                   # mesma seed
            elif k == pg.K_n:
                self.reset(seed=random.randrange(2**31))       # nova seed
            elif k == pg.K_d:
                self.show_sensors = not self.show_sensors
            elif k == pg.K_c:
                self.light.request_change()                    # pedido manual (teste)
            elif k == pg.K_UP:
                self.config.traffic_density = min(2.0, round(self.config.traffic_density + 0.1, 2))
            elif k == pg.K_DOWN:
                self.config.traffic_density = max(0.1, round(self.config.traffic_density - 0.1, 2))
            elif k in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                self.time_scale = min(8.0, self.time_scale * 2)
            elif k in (pg.K_MINUS, pg.K_KP_MINUS):
                self.time_scale = max(0.25, self.time_scale / 2)

    # ---- desenho
    _C_GRASS: Color = (46, 94, 60)
    _C_ROAD: Color = (60, 60, 64)
    _C_BOX: Color = (70, 70, 75)
    _C_LINE: Color = (235, 235, 235)
    _C_DASH: Color = (220, 190, 60)
    _C_TEXT: Color = (240, 240, 240)
    _LIGHT_COLORS = {
        LightState.RED: (231, 76, 60),
        LightState.YELLOW: (241, 196, 15),
        LightState.GREEN: (46, 204, 113),
    }
    _PHASE_LABELS = {
        Phase.NS_GREEN: "Norte/Sul VERDE",
        Phase.NS_YELLOW: "Norte/Sul AMARELO",
        Phase.ALL_RED_TO_EW: "Vermelho total (-> Leste/Oeste)",
        Phase.EW_GREEN: "Leste/Oeste VERDE",
        Phase.EW_YELLOW: "Leste/Oeste AMARELO",
        Phase.ALL_RED_TO_NS: "Vermelho total (-> Norte/Sul)",
    }

    def render(self) -> None:
        self._draw_road()
        for v in self.vehicles:
            self._draw_vehicle(v)
        self._draw_lights()
        self._draw_hud()
        if self.show_sensors:
            self._draw_sensor_panel()

    def _draw_road(self) -> None:
        pg, s, g = self._pg, self._screen, self.geometry
        s.fill(self._C_GRASS)
        pg.draw.rect(s, self._C_ROAD, (0, g.cy - g.half, g.width, g.box_size))
        pg.draw.rect(s, self._C_ROAD, (g.cx - g.half, 0, g.box_size, g.height))
        pg.draw.rect(s, self._C_BOX, (g.cx - g.half, g.cy - g.half, g.box_size, g.box_size))

        dash, gap = 16, 12
        for x in range(0, int(g.width), dash + gap):        # linha central horizontal
            if x + dash <= g.cx - g.half or x >= g.cx + g.half:
                pg.draw.line(s, self._C_DASH, (x, g.cy), (x + dash, g.cy), 2)
        for y in range(0, int(g.height), dash + gap):       # linha central vertical
            if y + dash <= g.cy - g.half or y >= g.cy + g.half:
                pg.draw.line(s, self._C_DASH, (g.cx, y), (g.cx, y + dash), 2)

        for d in Direction:
            pg.draw.rect(s, self._C_LINE, g.stop_line_rect(d))

        # rótulos de direção nas bordas
        labels = {
            Direction.NORTH: (g.cx + 40, 34),
            Direction.SOUTH: (g.cx - 90, g.height - 44),
            Direction.EAST: (g.width - 70, g.cy - 60),
            Direction.WEST: (8, g.cy + 44),
        }
        for d, pos in labels.items():
            s.blit(self._font_small.render(d.label.upper(), True, self._C_TEXT), pos)

    def _draw_vehicle(self, v: Vehicle) -> None:
        pg, s, g = self._pg, self._screen, self.geometry
        hx, hy = g.heading(v.origin)
        px, py = -hy, hx                                   # perpendicular
        fx, fy = g.world_pos(v.origin, v.position)
        rx, ry = g.world_pos(v.origin, v.position - v.length)
        hw = v.width / 2

        body = [(fx + px * hw, fy + py * hw), (fx - px * hw, fy - py * hw),
                (rx - px * hw, ry - py * hw), (rx + px * hw, ry + py * hw)]
        pg.draw.polygon(s, v.color, body)

        # faixa clara na FRENTE indica a direção do movimento
        sx, sy = fx - hx * 6, fy - hy * 6
        front = [(fx + px * hw, fy + py * hw), (fx - px * hw, fy - py * hw),
                 (sx - px * hw, sy - py * hw), (sx + px * hw, sy + py * hw)]
        pg.draw.polygon(s, (250, 250, 210), front)

        # luzes de freio
        if v.is_braking or v.is_stopped:
            for side in (-1, 1):
                bx = rx + hx * 2 + px * (hw - 3) * side
                by = ry + hy * 2 + py * (hw - 3) * side
                pg.draw.circle(s, (255, 30, 30), (int(bx), int(by)), 2)

        pg.draw.polygon(s, (255, 0, 0) if v.collided else (20, 20, 20), body, 2 if v.collided else 1)

    def _draw_lights(self) -> None:
        pg, s, g = self._pg, self._screen, self.geometry
        for d in Direction:
            state = self.light.state_for(d.group)
            x, y = g.light_pos(d)
            pg.draw.circle(s, (20, 20, 20), (int(x), int(y)), 11)
            pg.draw.circle(s, self._LIGHT_COLORS[state], (int(x), int(y)), 8)

    def _draw_hud(self) -> None:
        pg, s = self._pg, self._screen
        stats = [
            f"Vehicles: {self.vehicle_count}",
            f"Waiting: {self.waiting_count}",
            f"Collisions: {self.metrics.collisions}",
            f"Simulation Time: {self.time:.1f}s",
            f"Phase: {self._PHASE_LABELS[self.light.phase]}",
            f"Completed: {self.metrics.vehicles_completed}",
            f"Seed: {self.seed}   Density: {self.config.traffic_density:.1f}   Speed: {self.time_scale:g}x",
        ]
        panel = pg.Surface((360, 8 + 20 * len(stats)), pg.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        s.blit(panel, (8, 24))
        for i, line in enumerate(stats):
            s.blit(self._font.render(line, True, self._C_TEXT), (14, 28 + 20 * i))

        hint = ("ESPACO pausa | R reinicia | N nova seed | SETAS densidade | +/- velocidade | "
                "D sensores | C troca fase | ESC sai")
        s.blit(self._font_small.render(hint, True, self._C_TEXT), (8, self.geometry.height - 20))
        if self.paused:
            s.blit(self._font.render("PAUSADO", True, (255, 220, 80)), (self.geometry.width - 100, 30))

    def _draw_sensor_panel(self) -> None:
        pg, s = self._pg, self._screen
        data = self.get_sensor_data()
        lines = ["SENSOR DATA (debug)"]
        for d in Direction:
            lines.append(f"{d.label:<6} near={data.nearby_vehicles[d]:<2} wait={data.waiting_vehicles[d]:<2}")
        lines += [
            f"avg wait : {data.average_waiting_time:5.1f}s",
            f"avg speed: {data.average_speed:5.1f}px/s",
            f"phase    : {data.phase.value}",
            f"since chg: {data.time_since_phase_change:5.1f}s",
            f"can chg  : {data.can_change_phase}",
        ]
        w = 290
        panel = pg.Surface((w, 8 + 18 * len(lines)), pg.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        x0 = self.geometry.width - w - 8
        s.blit(panel, (x0, 60))
        for i, line in enumerate(lines):
            s.blit(self._font_small.render(line, True, self._C_TEXT), (x0 + 6, 64 + 18 * i))
