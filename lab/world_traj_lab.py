import numpy as np

from graph_search_lab import graph_search
from waypoint_traj_lab import WaypointTraj

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.125, 0.125, 0.125])
        self.margin = 0.30

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.

        self.points = self.path[0,:]   # Include the first point from path

        # SHANE LEVY/WILL YANG gave me the idea to make heavy use of of the path_collisions() method from world to prune
        # points. We will loop over points in the path, for each point we continue to delete points as long as they
        # don't result in a collision
        current_idx = 0
        for idx in np.arange(1,self.path.shape[0]):
            # Check each idx, if collision does not occur then add it to the self.points
            collision_pts = world.path_collisions(np.array([self.path[current_idx,:],self.path[idx,:]]),self.margin+0.05)
            if collision_pts.size != 0: # A collision occurs
                self.points = np.vstack((self.points, self.path[idx,:]))  # Add points up to that collision
                current_idx = idx  # Reset idx to the last point before collision!

        self.points = np.vstack((self.points, self.path[-1,:]))  # Add the last point


        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

        if self.points.size > 3:
            # i.e. if we're giving it more than one point run WaypointTraj. Otherwise, we could run hover_traj.
            self.my_traj = WaypointTraj(self.points)

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE

        flat_output = self.my_traj.update(t)  # Redirect to waypoint_traj from project 1 (lazy implementation)

        return flat_output
