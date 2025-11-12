## DESCRIPTION
This is the code for the final project of AA228/CS238 in 2025.

The objective of this project is to enhance the decision-making capabilities of autonomous
vehicles at intersections by developing a Partially Observable Markov Decision Process
(POMDP) model. The aim is to enable the autonomous vehicle to safely and efficiently
drive through intersections by accounting for the positions, velocities, and headings of other
agents, as well as the state of traffic lights. Success in this project will be measured by the
vehicle’s ability to make optimal decisions that minimize collision risk and travel time and
maximize the comfort while adhering to traffic rules.
This problem is particularly interesting because intersections are complex and dynamic
environments where multiple agents interact with each other. Effective decision-making in
such scenarios is crucial for the safe deployment of autonomous vehicles. As a reference,
Zhan et al. (2019) provides a behavior datasets of traffic participants in highly interactive
driving scenarios at https://interaction-dataset.com/.

## Decision Making
The problem is a sequential decision-making problem where an autonomous vehicle must
continuously make decisions at intersections. The environment consists of other agents
(vehicles, bikes, pedestrians, etc.) and traffic lights, both of which influence the autonomous
vehicle’s actions.

We can model this problem as a POMDP, which includes the following components:
* States: The states of the environment, including the positions, velocities, and headings
of all agents, and the states of the traffic lights at the intersection.
* Actions: The set of possible maneuvers the autonomous vehicle can take, such as
accelerating, decelerating, stopping, turning left, turning right, or going straight. The
actions can be classified into longitudinal and lateral behaviors.
* Transitions: The probabilistic model describing how the environment evolves in re-
sponse to the vehicle’s actions, influenced by the behavior of other agents and traffic
light changes.
* Rewards: The objective function to be maximized, which could include factors such
as safety (collision avoidance), efficiency (travel time), comfort(jerk), and compliance
with traffic regulations.
* Observations: The partial observations available to the vehicle, such as sensor readings
that provide other agents’ states and the traffic light status.

## Sketches of Solution
We start this with SARSA: $Q(s,a)←Q(s,a)+\alpha (r + \gamma Q(s', a') - Q(s, a))$.

