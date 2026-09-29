# Traffic Lights AI

> **Pre-Alpha — v0.1**

Traffic Lights AI is an experimental project that explores whether a neural network can **learn how to control traffic lights efficiently** through simulation.

The idea is simple:

> Instead of programming a fixed rule for when a traffic light should change, let an AI learn a strategy by observing traffic and being rewarded for making better decisions.

---

## The Idea

The project starts with a simple four-direction intersection:

```text
                 ↓
                 │
                 │
        → ───────┼─────── →
                 │
                 │
                 ↑
```

Vehicles continuously arrive at the intersection.

The AI observes the traffic situation and decides whether the current traffic-light phase should remain active or change.

For example, it may learn that:

* There are many cars waiting in one direction.
* There are almost no cars in the other direction.
* Keeping the current light green is no longer useful.
* Changing the phase would reduce waiting time.

The AI is not given a predefined traffic strategy.

It has to **discover one through training**.

---

## How the AI Learns

The first version uses a neural network combined with an evolutionary training process.

Multiple networks are created and tested in the traffic simulation.

Their performance is measured using factors such as:

* Number of collisions
* Vehicle waiting time
* Traffic flow
* Number of unnecessary stops

Better-performing networks are selected and mutated to create the next generation.

```text
Neural Networks
       ↓
Traffic Simulation
       ↓
Performance Evaluation
       ↓
Selection
       ↓
Mutation
       ↓
New Generation
       ↓
Repeat
```

The objective is to see whether the population progressively discovers better traffic-light strategies.

---

## What the AI Controls

The AI controls the **traffic-light phases**, not the vehicles.

For the initial intersection, traffic is divided into two compatible directions:

```text
North / South
      ↕
East / West
      ↔
```

The neural network can decide whether to:

```text
Keep the current phase
        or
Request a phase change
```

A separate safety system controls the actual transition between green, yellow, and red.

This prevents the neural network from creating invalid combinations such as allowing conflicting directions to move simultaneously.

---

## Why Start So Small?

The long-term goal is much larger.

Eventually, the project could explore:

```text
        Intersection
             ↓
    Multiple Intersections
             ↓
      Traffic Networks
             ↓
       Larger Cities
```

However, the first question needs to be answered first:

> **Can a relatively small neural network actually learn a useful traffic-light strategy?**

The v0.1 version exists to answer that question.

If the basic system works, complexity can be added gradually.

---

## v0.1

The first version focuses on:

* One four-direction intersection
* Basic vehicle simulation
* Traffic lights
* Traffic sensors
* Neural network
* Evolutionary training
* Multiple generations
* Fitness evaluation
* Collision detection
* Waiting-time measurement
* Basic visualization
* Saving and loading trained networks

The first version intentionally does **not** attempt to simulate an entire city.

---

## Measuring Success

The AI is not considered successful simply because its fitness score increases.

Its behavior should be compared against simple strategies, such as:

* Random traffic-light control
* Fixed-time traffic lights
* Evolved neural-network control

The goal is to determine whether the neural network can produce a measurable improvement over simple approaches.

---

## Long-Term Goal

The project will progressively increase the complexity of the environment.

Possible future stages include:

### More complex intersections

* Turning lanes
* Different traffic-light configurations
* More traffic patterns
* More sensors

### Procedural environments

* Randomly generated intersections
* User-created intersections
* Different road layouts

### Multiple intersections

```text
      A ───── B
      │       │
      │       │
      C ───── D
```

Each intersection could have its own neural network, allowing local decisions to affect traffic throughout the network.

### More advanced AI

Future versions may also experiment with:

* Traditional reinforcement-learning algorithms
* Multi-agent learning
* Larger neural networks
* GPU training
* Knowledge transfer between networks

---

## Project Philosophy

Traffic Lights AI is primarily a **learning and experimentation project**.

The goal is not to immediately build a production-ready traffic-management system.

The goal is to progressively investigate:

**Can an AI learn a complex real-world-inspired behavior from a simulated environment?**

Starting with a simple intersection makes it possible to understand the AI, the simulation, the training process, and the results before increasing the complexity of the problem.

---

## Technology

Current development:

* Python
* NumPy
* Pygame
* Local training

The neural-network and evolutionary systems are being developed as part of the project to provide a better understanding of how they work internally.

---

## Status

**Pre-Alpha — v0.1**

The project is actively under development.

The current version is focused on establishing the fundamental learning loop:

```text
Observe
   ↓
Decide
   ↓
Simulate
   ↓
Evaluate
   ↓
Improve
   ↓
Repeat
```

---

## Roadmap

* [ ] Basic traffic simulation
* [ ] Vehicle simulation
* [ ] Four-direction intersection
* [ ] Traffic-light system
* [ ] Neural network
* [ ] Evolutionary training
* [ ] Fitness system
* [ ] Collision detection
* [ ] Visualization
* [ ] Save/load trained networks
* [ ] Baseline comparisons
* [ ] More complex intersections
* [ ] Procedural environments
* [ ] Multiple intersections
* [ ] Larger traffic networks

---

## License

This project is licensed under the MIT License.

See [`LICENSE`](LICENSE) for details.

---

## Author

**Luca Filippi**

Software Engineering student and software development enthusiast.

GitHub: [LucaFilippi](https://github.com/LucaFilippi)
