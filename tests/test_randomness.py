import unittest

from src.simulation import Simulation, SimulationConfig
from src.traffic_light import Phase
from src.vehicle import Direction, SignalDecision, Vehicle


class RandomnessTests(unittest.TestCase):
    def test_direction_bias_changes_spawn_interval(self):
        cfg = SimulationConfig(seed=7, direction_bias=(3.0, 1.0, 0.5, 1.5))
        sim = Simulation(cfg)

        north = sim._spawn_interval_for(Direction.NORTH, 1.0)
        east = sim._spawn_interval_for(Direction.EAST, 1.0)

        self.assertLess(north, east)

    def test_yellow_runner_keeps_moving_through_yellow(self):
        sim = Simulation(SimulationConfig(seed=2))
        sim.light.phase = Phase.NS_YELLOW

        v = Vehicle(
            id=1,
            origin=Direction.NORTH,
            destination=Direction.SOUTH,
            position=90.0,
            speed=50.0,
            max_speed=75.0,
            acceleration=60.0,
            deceleration=180.0,
            yellow_runner=True,
        )

        target = sim._target_speed(
            v,
            leader=None,
            stop_s=sim.geometry.stop_s(v.origin),
            light_state=sim.light.state_for(v.origin.group),
        )

        self.assertIs(v.signal_decision, SignalDecision.GO)
        self.assertGreaterEqual(target, 50.0)


if __name__ == "__main__":
    unittest.main()
