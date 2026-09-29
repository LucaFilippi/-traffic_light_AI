"""Ponto de entrada do Traffic Lights AI (pré-alpha v0.1)."""

import argparse

from src.simulation import Simulation, SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic Lights AI - simulador base")
    parser.add_argument("--seed", type=int, default=None, help="seed aleatória (reprodutível)")
    parser.add_argument("--density", type=float, default=None, help="densidade do tráfego (ex.: 0.5)")
    parser.add_argument("--headless", type=float, metavar="SEGUNDOS", default=None,
                        help="roda sem janela por N segundos simulados e imprime um resumo")
    args = parser.parse_args()

    config = SimulationConfig()
    if args.seed is not None:
        config.seed = args.seed
    if args.density is not None:
        config.traffic_density = args.density

    simulation = Simulation(config)
    if args.headless is not None:
        simulation.run_headless(args.headless)
        print(simulation.summary())
    else:
        simulation.run()


if __name__ == "__main__":
    main()
