from heapq import heapify, heappop, heappush  # Recommended.
import numpy as np

from flightsim.world import World
from proj1_2.code.occupancy_map import OccupancyMap # Recommended.

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

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    goal_found = False
    delta = [-1, 0, 1]

    # nei=[]
    # for i in delta:
    #     for j in delta:
    #         for k in delta:
    #             if (i, j, k) != (0, 0, 0): nei.append([i,j,k])
    # neighbours=np.array(nei)
    neighbours=np.array([[-1, -1, -1],
                           [-1, -1,  0],
                           [-1, -1,  1],
                           [-1,  0, -1],
                           [-1,  0,  0],
                           [-1,  0,  1],
                           [-1,  1, -1],
                           [-1,  1,  0],
                           [-1,  1,  1],
                           [ 0, -1, -1],
                           [ 0, -1,  0],
                           [ 0, -1,  1],
                           [ 0,  0, -1],
                           [ 0,  0,  1],
                           [ 0,  1, -1],
                           [ 0,  1,  0],
                           [ 0,  1,  1],
                           [ 1, -1, -1],
                           [ 1, -1,  0],
                           [ 1, -1,  1],
                           [ 1,  0, -1],
                           [ 1,  0,  0],
                           [ 1,  0,  1],
                           [ 1,  1, -1],
                           [ 1,  1,  0],
                           [ 1,  1,  1]])

    parent = {start_index: None}


    if astar:

        start_center = occ_map.index_to_metric_center(start_index)
        goal_center = occ_map.index_to_metric_center(goal_index)
        distance = {start_index: 0}
        f = {start_index: np.linalg.norm(start_center-goal_center,2)}
        h = [(f[start_index], start_index)]
        heapify(h)

        while h:
            (e, u) = heappop(h)
            g = distance[u]
            if u == goal_index:
                goal_found = True
                break

            for i in neighbours:
                v = tuple(i + np.array(u))
                if occ_map.is_valid_index(v) and not (occ_map.is_occupied_index(v)):
                    q = occ_map.index_to_metric_center(u)
                    p = occ_map.index_to_metric_center(v)
                    w = np.linalg.norm(q - p, 2)
                    if (v not in distance) or ((g + w) < distance[v]):
                        f[v] = g + w + np.linalg.norm(p - goal_center,2)
                        distance[v] = g + w
                        heappush(h, (f[v], v))
                        parent[v] = u

    else:
        distance = {start_index: 0}
        h = [(distance[start_index], start_index)]
        heapify(h)

        while h:
            (g, u)=heappop(h)

            if u == goal_index:
                goal_found=True
                break

            for i in neighbours:
                v=tuple(i+np.array(u))
                if occ_map.is_valid_index(v) and not(occ_map.is_occupied_index(v)):
                    q = occ_map.index_to_metric_center(u)
                    p = occ_map.index_to_metric_center(v)
                    w = np.linalg.norm(q - p, 2)
                    if (v not in distance) or ((g + w) < distance[v]):
                        distance[v] = g + w
                        heappush(h, (distance[v], v))
                        parent[v] = u

    if goal_found:

        path=[]
        path.append(goal)
        path.append(occ_map.index_to_metric_center(goal_index))

        par=parent[goal_index]
        while par:
            path.append(occ_map.index_to_metric_center(par))
            par=parent[par]

        path.append(start)

        return np.flip(path,axis=0)
    else:
        return None
