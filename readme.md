# Traffic Lights AI

> **Pre-Alpha — v0.1**

An experimental artificial intelligence project focused on teaching neural networks to control traffic lights through reinforcement learning.

The long-term goal is to create neural networks capable of learning how to coordinate traffic lights across an arbitrary number of intersections, whether those intersections are procedurally generated or manually designed by the user.

This project is primarily an AI, simulation, and software engineering experiment designed to explore how neural networks can learn complex traffic-control behaviors from interaction with a simulated environment.

---

## Objective

The main objective of Traffic Lights AI is to develop an artificial agent capable of learning how to control traffic signals without being explicitly programmed with a predefined traffic-light strategy.

The AI should learn through experimentation:

* When to keep a traffic light green
* When to switch it to yellow
* When to switch it to red
* How long a signal should remain in each state
* How to react to different traffic densities
* How to react to pedestrians
* How to minimize unnecessary stops
* How to avoid dangerous signal configurations
* How to coordinate multiple intersections

The ultimate goal is to allow a trained neural network to control traffic lights across increasingly complex road networks.

---

# Concept

The simulation represents a traffic environment containing:

* Roads
* Intersections
* Traffic lights
* Vehicles
* Pedestrians
* Traffic flow
* Sensors
* Collisions
* Traffic-light states

Each intersection is controlled by an independent neural network, referred to as its **"brain"**.

A brain controls the traffic lights belonging to its intersection.

Depending on the intersection configuration, a brain may control:

* 2 traffic lights
* 3 traffic lights
* 4 traffic lights

More complex intersections may be supported as the project evolves.

Multiple intersections can coexist in the same simulation, allowing individual neural networks to interact indirectly through the traffic environment.

---

# Reinforcement Learning

Traffic Lights AI uses reinforcement learning as its fundamental learning approach.

The neural network does not receive a predefined sequence telling it exactly when to change each traffic light.

Instead, it interacts with the environment:

```text
        Environment
             ↓
       Traffic State
             ↓
       Neural Network
             ↓
           Action
             ↓
       Traffic Lights
             ↓
         Simulation
             ↓
          Reward
             ↓
       Network Update
             ↓
          Repeat
```

The network is rewarded for desirable traffic behavior and penalized for undesirable behavior.

The objective is therefore not simply to keep traffic moving.

The AI must learn to balance multiple competing factors.

---

# Neural Network

One of the central ideas of the project is that the user should be able to experiment with different neural-network architectures.

The user can configure the number of hidden layers and the number of neurons within those layers.

This allows different network architectures to be trained and compared.

There is intentionally no single predefined "correct" architecture.

The project is designed to investigate how different network structures affect learning behavior.

The practical architectural limit is determined by the available computational resources.

---

# Inputs

The neural network receives information about the current state of the intersection.

The planned input information includes:

* Vehicle speed
* Number of vehicles approaching a traffic light
* Number of vehicles approaching other controlled signals
* Pedestrian crossing requests
* Time a vehicle has been waiting at a red light
* Time required for a vehicle traveling at the road speed limit to detect a yellow light and safely brake
* Number of collisions caused by incompatible simultaneous signal states

These inputs provide the neural network with information about both the immediate traffic situation and the consequences of previous decisions.

---

# Outputs

The traffic-light controller has three fundamental actions:

```text
0 → Red
1 → Yellow
2 → Green
```

The network therefore determines the desired state of the controlled traffic signal.

The simulation itself is responsible for enforcing traffic rules and safety constraints.

---

# Reward System

The reward system is designed around three major objectives:

### 1. Safety

Collisions are heavily penalized.

The AI should learn to avoid traffic-light configurations that allow conflicting traffic movements to occur simultaneously.

### 2. Waiting Time

Vehicles waiting unnecessarily at red lights generate a negative reward.

For example, keeping a signal red while another direction has a green signal but no vehicles are present should be considered inefficient.

### 3. Vehicle Efficiency

The system rewards traffic that can move through an intersection without unnecessary braking and acceleration.

Reducing unnecessary stops should result in:

* Lower waiting time
* Smoother traffic flow
* Lower simulated fuel consumption

The reward system is therefore intended to encourage the AI to find efficient solutions rather than simply maximizing the number of vehicles passing through an intersection.

---

# Training

Training occurs through repeated simulation.

A generation can contain many individuals, potentially more than 50 neural networks.

Each individual is evaluated by running its neural network inside the traffic simulation for a user-defined amount of simulated time.

For example:

```text
Generation
    │
    ├── Neural Network 01
    ├── Neural Network 02
    ├── Neural Network 03
    ├── ...
    └── Neural Network N
             │
             ↓
       Traffic Simulation
             │
             ↓
          Evaluation
             │
             ↓
          Fitness
             │
             ↓
       Selection
             │
             ↓
          Mutation
             │
             ↓
      Next Generation
```

The user may determine how long each network should be trained/evaluated.

Training can potentially run for extended periods, including simulations lasting up to approximately one hour or more depending on the configuration and available hardware.

Multiple individuals may also be simulated in parallel when computational resources allow it.

---

# Evolution and Mutation

The project includes an evolutionary component in which neural networks can be selected based on their performance and mutated to produce new candidates.

Better-performing networks have a greater opportunity to reproduce their architecture/parameters into the next generation.

Mutation introduces variation into the population.

This allows the system to explore different solutions instead of repeatedly reproducing the exact same network.

The project is therefore also an experiment in combining:

* Neural networks
* Reinforcement learning
* Evolutionary optimization
* Procedural simulation

---

# Simulation

The traffic simulation is designed to be configurable.

Users should eventually be able to create or configure intersections including:

* Number of roads
* Number of traffic lanes
* Traffic-light configuration
* Allowed turning directions
* Traffic density
* Vehicle placement
* Pedestrian placement
* Traffic-light behavior
* Intersection topology

Intersections may also be randomly generated.

This allows the AI to be tested against environments it has not necessarily encountered before.

---

# Multiple Intersections

A major long-term objective is to move beyond isolated intersections.

The simulation is intended to support an arbitrary number of intersections.

Conceptually:

```text
       Intersection A
             │
             │
      ┌──────┴──────┐
      │             │
Intersection B   Intersection C
      │             │
      └──────┬──────┘
             │
       Intersection D
```

Each intersection can have its own neural network.

The networks do not need to directly communicate with one another.

Instead, their decisions affect the shared traffic environment.

This creates a more complex problem where local decisions can have consequences for other intersections.

---

# User Configuration

The user is not intended to manually control traffic lights during normal operation.

Instead, the user configures the environment.

Possible configuration options include:

* Building an intersection manually
* Generating an intersection randomly
* Adding vehicles
* Adding pedestrians
* Configuring traffic
* Choosing the neural-network architecture
* Choosing the number of hidden layers
* Choosing the number of neurons
* Defining training duration
* Running multiple individuals simultaneously
* Saving trained neural networks
* Loading previously trained networks

The ability to save and reload networks is important for long-running experiments.

A user should be able to train a network for several hours, stop the program, and continue training later.

---

# Evaluation

The primary performance indicators are:

### Safety

How many collisions occur?

### Waiting Time

How long do vehicles remain unnecessarily stopped at red lights?

### Traffic Flow

How efficiently do vehicles pass through the intersection?

### Vehicle Efficiency

How much unnecessary braking and acceleration occurs?

### Simulated Fuel Consumption

The simulation may use vehicle acceleration and braking behavior as an approximation for fuel consumption.

The objective is to encourage smooth traffic flow rather than simply maximizing vehicle throughput.

---

# Why This Project?

Traffic Lights AI is primarily a personal research and engineering project.

The project is being developed to improve practical knowledge in:

* Artificial intelligence
* Neural networks
* Reinforcement learning
* Evolutionary algorithms
* Simulation
* Optimization
* Software architecture
* Python
* Computational performance

A secondary long-term objective is to investigate whether trained networks could eventually be reused in other simulations or games.

A trained neural network can potentially be exported and integrated into another application without requiring the entire training process to be repeated.

---

# Current Status

## v0.1 — Pre-Alpha

The project is currently in the **Pre-Alpha 0.1** stage.

At this stage, the primary focus is building the underlying simulation, neural-network infrastructure, training system, reward system, and experimentation environment.

The architecture is expected to change significantly during development.

The current version should therefore be considered experimental and unstable.

---

# Technology

Current development is being performed locally using:

* **Python**
* Neural-network implementation under development
* Traffic simulation under development
* Local CPU/GPU computation depending on the implementation

Specific libraries and frameworks may change during the Pre-Alpha phase as different approaches are evaluated.

---

# Project Structure

The project structure is expected to evolve as development progresses.

A planned architecture may include:

```text
traffic_light_AI/
│
├── src/
│   ├── simulation/
│   ├── neural_network/
│   ├── reinforcement_learning/
│   ├── training/
│   ├── traffic/
│   └── visualization/
│
├── models/
│   └── trained_networks/
│
├── experiments/
│
├── tests/
│
├── docs/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Roadmap

The roadmap is intentionally flexible because the project is still in its experimental phase.

### v0.1 — Pre-Alpha

* [x] Initial repository
* [x] MIT License
* [x] Readme file
* [ ] Traffic simulation
* [ ] Vehicle simulation
* [ ] Traffic lights
* [ ] Intersection system
* [ ] Sensors
* [ ] Neural-network system
* [ ] Reinforcement-learning environment
* [ ] Reward system
* [ ] Training system
* [ ] Mutation/evolution system
* [ ] Multiple individuals per generation
* [ ] Save neural networks
* [ ] Load neural networks
* [ ] Graphical interface

### Future Versions

Potential future developments include:

* [ ] Procedural intersection generation
* [ ] User-created intersections
* [ ] Multiple interconnected intersections
* [ ] Larger traffic networks
* [ ] More complex pedestrian behavior
* [ ] More realistic vehicle dynamics
* [ ] Improved reward functions
* [ ] GPU acceleration
* [ ] Distributed training
* [ ] Network visualization
* [ ] Training statistics
* [ ] AI-vs-AI experiments
* [ ] Export trained neural networks
* [ ] Integration experiments with other simulations and games

---

# Versioning

The project uses development versions during the experimental phase.

Current version:

**v0.1-pre-alpha**

This version represents an experimental milestone rather than a stable release.

---

# License

This project is licensed under the MIT License.

See the [`LICENSE`](LICENSE) file for details.

---

# Author

**Luca Filippi**

Software Engineering student at PUC-Campinas and software development enthusiast. 

GitHub: [LucaFilippi](https://github.com/LucaFilippi)

---

## Project Status

**Experimental — Pre-Alpha**

The project is actively under development.

Expect significant changes to the architecture, algorithms, simulation rules, and neural-network implementation as experimentation continues.
