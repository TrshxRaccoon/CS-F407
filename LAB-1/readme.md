---

**Chosen Algorithm: A\* (A-Star) Search**

- **How it works:** A\* evaluates nodes using the cost function $f(n) = g(n) + h(n)$, where:
- $g(n)$ is the exact cost of the path from the starting point to node $n$.
- $h(n)$ is an estimated heuristic cost to move from node $n$ to the goal. In grid navigation limited to 4 directions (Up, Down, Left, Right), we use **Manhattan Distance**: $h(n) = \vert{}x_n - x_g\vert{} + \vert{}y_n - y_g\vert{}$.

**Why A\* is appropriate for this warehouse scenario:**

- **Optimality & Guarantee:** A\* guarantees finding the shortest collision-free path available (assuming unit step cost and an admissible heuristic that never overestimates the distance to the goal).
- **Efficiency:** Unlike Uniform Cost Search (Dijkstra's) or Breadth-First Search (BFS)—which blindly expand in all directions—A\* uses the heuristic $h(n)$ to direct its exploration towards the goal, drastically reducing unnecessary node expansions around obstacles.
- **Flexibility for Warehouses:** It easily scales to dynamic grid sizes, non-uniform movement costs (e.g., fast lanes vs. slow zones), and complex obstacle shapes typical of real-world automated guided vehicle (AGV) pathfinding.
