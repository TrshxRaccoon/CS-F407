import heapq

def a_star_warehouse(grid):
    # Locate Start (S) and Goal (G)
    start = None
    goal = None
    rows = len(grid)
    cols = len(grid[0])
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
                
    if not start or not goal:
        return "Path does not exist"
    
    # Manhattan distance heuristic function
    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])
    
    # Priority queue stores tuples of: (f_score, (row, col))
    open_set = []
    heapq.heappush(open_set, (heuristic(start), start))
    
    # Track the actual cost to reach each node (g_score)
    g_score = {start: 0}
    
    # Track paths: came_from[node] = parent_node
    came_from = {}
    
    # Movement directions: Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while open_set:
        current_f, current = open_set.pop(0)  # Extract node with lowest f_score
        
        # Goal test
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        # Explore neighbors
        r, c = current
        for dr, dc in directions:
            neighbor = (r + dr, c + dc)
            nr, nc = neighbor
            
            # Check boundaries and obstacles
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
                tentative_g = g_score[current] + 1  # Cost of 1 step
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
    return "Path does not exist"

def print_solution(grid, path):
    if isinstance(path, str):
        print(path)
        return
        
    print(f"Path found with length {len(path) - 1} steps:\n")
    # Make a copy of the grid for visual printing
    grid_copy = [list(row) for row in grid]
    
    # Mark path with '*' except start and goal
    for r, c in path:
        if grid_copy[r][c] not in ('S', 'G'):
            grid_copy[r][c] = '*'
            
    for row in grid_copy:
        print(" ".join(row))

# --- Example Usage ---
warehouse_map = [
    ['S', '.', '.', '#', '.', '.'],
    ['.', '#', '.', '#', '.', '.'],
    ['.', '#', '.', '.', '.', '.'],
    ['.', '.', '#', '#', '.', '.'],
    ['#', '.', '.', '.', '#', 'G']
]

path_result = a_star_warehouse(warehouse_map)
print_solution(warehouse_map, path_result)