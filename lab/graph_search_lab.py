from heapq import heappush, heappop, heapify  # Recommended.
import numpy as np

from flightsim.world import World
from occupancy_map_lab import OccupancyMap # Recommended.
from math import sqrt
import itertools

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
    """

    def heuristic(X1, X2):
        # Function takes in two points (np.array: (3,)) and determines the L2 norm between both points.
        # L2norm(X1, X2) = sqrt( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 )
        # h = np.linalg.norm(X1 - X2)
        h = sqrt( (X1[0]-X2[0])**2+(X1[1]-X2[1])**2+(X1[2]-X2[2])**2 )
        return h

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    # Initialize the priority queue:
    # priority_queue = []
    # g = 0
    # heappush(priority_queue, (g, start_index))
    # heappush(priority_queue, (np.inf,goal_index))

    priority_queue = PQ()
    g = 0
    priority_queue.add_task(start_index,g)
    priority_queue.add_task(goal_index,np.inf)

    # Set up a parent node tree, JOSH REMBA and GREG CAMPBELL suggested a dictionary was most suitable for this.
    # The dictionary holds all parent nodes. It has the following structure:
    # {'neighbor idx' : [g value, neighbor_idx, parent_idx]}
    # This will be updated in the while loop.. a parent will be added to the dict and updated accordingly
    parent_nodes = {str(start_index): [0, start_index, start_index]}

    # GREG CAMPBELL also had the genius idea of creating a list of possible neighbors (x,y,z) OUTSIDE the while loop
    # to mitigate the number of nested loops. As a result in the while loop we only need one loop instead of three.
    neighbors = []
    for x in (-1,0,1):
        for y in (-1,0,1):
            for z in (-1,0,1):
                # Include an if statement such that neighbors doesn't include your current point
                if (x,y,z) != (0,0,0):
                    neighbors.append((x,y,z))

    path_found = False      # bool for finding a path and exiting while loop

    # WILL YANG had the idea to use a set to keep track of expanded nodes (from priority queue), so we can avoid
    # expanding those in the future.
    expanded_nodes = set()

    # Keep a number of the expanded nodes
    nodes_expanded = 0

    while not path_found:
        # First pop out the smallest g from priority queue
        # current_node = heappop(priority_queue)  # Before fixing heapq
        current_node = priority_queue.pop_task()
        current_loc = occ_map.index_to_metric_center(current_node)

        # Add this to the list of expanded nodes
        expanded_nodes.add(current_node)

        # Keep count of expanded nodes to confirm astar works.
        nodes_expanded += 1

        if current_node == goal_index:
            # Once our current node is the goal position we've finished
            path_found = True
            break

        # Now loop over the 26 neighbors x,y,z
        for neighbor in neighbors:
            # For each neighbor estimate the g(neighbor) = g(current_node) + c(current_node,neighbor)

            # First extract neighbor node index
            neighbor_node = (current_node[0] + neighbor[0], current_node[1] + neighbor[1],
                             current_node[2] + neighbor[2])

            if occ_map.is_valid_index(neighbor_node) and not occ_map.is_occupied_index(neighbor_node) \
                    and neighbor_node not in expanded_nodes:
                # Make sure to check if the index is IN the occ_map and that it is NOT occupied. Also check that the
                # node hasn't already been expanded.

                # Determine neighbor location in world
                neighbor_loc = occ_map.index_to_metric_center(neighbor_node)

                # Find cost to come based on Euclidean distance
                c = heuristic(current_loc, neighbor_loc)

                # Compute d, a sum of the current cost at that node and the cost to come
                d = parent_nodes[str(current_node)][0] + c
                if not astar:
                    # If not using astar, f is just the value of d
                    f = d
                else:
                    # If using astar, we add on a heuristic to the goal.
                    f = d + heuristic(current_loc,goal)

                if str(neighbor_node) not in parent_nodes.keys() or d < parent_nodes[str(neighbor_node)][0]:
                    # If the neighbor node isn't saved in the parent dict, or if its d value now less than the
                    # previously saved value (i.e. we found a shorter path to that node)
                    # Save the neighbor node along with its cost and its parent.
                    parent_nodes[str(neighbor_node)] = [d, neighbor_node, current_node]

                    # Push the min neighbor and corresponding cost g into the priority queue
                    # heappush(priority_queue, (d,neighbor_node))
                    priority_queue.add_task(neighbor_node,f)

    print("Nodes Expanded: "+str(nodes_expanded))
    if len(priority_queue.pq) == 0:
        # If the priority queue is empty then we know no path exists
        path = None
    else:
        # Now we have must construct our path using the parent_nodes dictionary
        path = np.array(occ_map.index_to_metric_center(goal_index)).reshape(1,3)  # Initialize with the end goal
        current_path_node = goal_index

        while np.any(occ_map.index_to_metric_center(start_index) != path[0]):
            # Loop up through the tree starting from goal and ending at start
            next_node = parent_nodes[str(current_path_node)][2]
            next_loc = occ_map.index_to_metric_center(next_node)
            path = np.vstack((next_loc,path))

            current_path_node = next_node

        # Finally append start and end positions to our path
        path = np.vstack((start, path))
        path = np.vstack((path, goal))
    return path

class PQ:
# TAKEN FROM https://docs.python.org/3/library/heapq.html
# This class wraps around heapq to add more functionality to eat to make it a proper priority queue.

    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self,task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')