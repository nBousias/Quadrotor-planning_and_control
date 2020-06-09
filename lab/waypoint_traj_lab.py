import numpy as np
from scipy.interpolate import CubicSpline

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE

        # Target acceleration between each waypoint:
        self.const_a = 1.5 # m/s^2

        # Fit 3D lines in between each waypoint

        self.waypoints = points

        N = self.waypoints.shape[0]
        dist = np.zeros((N-1,))
        self.unit_vec = np.zeros((N-1,3))
        self.times = np.zeros((N,))
        for i in range(N-1):
            # Loop thru N waypoints and get distance between each point.
            dist[i] = np.linalg.norm(np.array([points[i][0] - points[i+1][0],
                                               points[i][1] - points[i+1][1],
                                               points[i][2] - points[i+1][2]]))

            # Also save the unit vector corresponding to each distance
            if dist[i] != 0:
                self.unit_vec[i,:] = (points[i+1,:] - points[i,:])/dist[i]
            else:
                self.unit_vec[i,:] = np.zeros((3,))

            # Compute times between each node given a constant acceleration
            self.times[i+1] = self.times[i] + np.sqrt(4*dist[i]/self.const_a)



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

        if t < self.times[-1]:   # Make sure whether or not we've theoretically reached the waypoint
            if np.any(t == self.times):  # Avoid edge case where we're the below logical condition fails
                # If t is part of our times, it means we should be at a waypoint, so find that waypoint and set zero
                # acceleration, speed, and position = waypoint.
                seg_idx = np.where(t == self.times)[0][0]
                x = self.waypoints[seg_idx, :]
            else:  # This case occurs when we're between segments
                seg_idx = np.where((t >= self.times[0:-1]) * (t <= self.times[1:]))[0][0]  # returns the index corresponding to the start of the segment!
                tstar = (self.times[seg_idx+1] - self.times[seg_idx])/2  # midway point for our particular segment

                # Now check which half of the segment we're in
                if (t-self.times[seg_idx]) <= tstar:
                    # If t < half segment time, then we're in the first portion
                    a_t = self.const_a
                    v_t = self.const_a*(t-self.times[seg_idx])
                    p_t = (1/2)*self.const_a*(t-self.times[seg_idx])**2
                else:
                    # Otherwise we're in the second portion of the segment, math gets uglier bc of initial conditions
                    a_t = -self.const_a
                    v_t = self.const_a*tstar - self.const_a*(t-self.times[seg_idx] - tstar)
                    p_t = (1/2)*self.const_a*(tstar**2) - (1/2)*self.const_a*(t-self.times[seg_idx] - tstar)**2 + self.const_a*tstar*(t - self.times[seg_idx] - tstar)

                # We can now back out our x_ddot, x_dot, and x by projecting our a_t, v_t, and p_t from unit_vect to inertial frame
                unit_vec_seg = self.unit_vec[seg_idx,:]
                x_ddot = a_t*unit_vec_seg
                x_dot = v_t*unit_vec_seg
                x = self.waypoints[seg_idx] + p_t*unit_vec_seg
        else:
            # If we're at the waypoint, simply supply the goal waypoint for x and don't set values for x_dot, x_ddot
            x = self.waypoints[-1,:]

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
