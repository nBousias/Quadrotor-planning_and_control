import numpy as np
from proj1_3.code.graph_search import graph_search
from scipy import interpolate
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from flightsim.axes3ds import Axes3Ds


def coef(x0,x1,t0,t1,dx0=0,ddx0=0,dx1=0,ddx1=0):
    # dt=t1-t0
    # H=np.array([[720*T**5, 360*T**4, 120*T**3],
    #             [360*T**4, 192*T**3, 72*T**2],
    #             [120*T**3, 72*T**2 , 36*T]])
    L=np.linalg.inv(np.array([[t0**5, t0**4, t0**3, t0**2, t0, 1],
                              [5*t0**4, 4*t0**3, 3*t0**2, 2*t0, 1, 0],
                              [20*t0**3, 12*t0**2, 6*t0, 2, 0, 0],
                              [t1 ** 5, t1 ** 4, t1 ** 3, t1 ** 2, t1, 1],
                              [5 * t1 ** 4, 4 * t1 ** 3, 3 * t1 ** 2, 2 * t1, 1, 0],
                              [20 * t1 ** 3, 12 * t1 ** 2, 6 * t1, 2, 0, 0]]))

    coefficients = L @ np.array([x0, dx0, ddx0, x1, dx1, ddx1]).T
    return coefficients

def min_jerk_zeros(points,t):
    N=t.shape[0]
    c=np.zeros((N-1,6,3))
    for i in range(N-1):
        c[i, :, 0]  = coef(points[i, 0], points[i + 1, 0], t[i], t[i + 1])
        c[i, :, 1] = coef(points[i, 1], points[i + 1, 1], t[i], t[i + 1])
        c[i, :, 2] = coef(points[i, 2], points[i + 1, 2], t[i], t[i + 1])
    return c


def resample_points(world,points,margin):
    ind=[]
    N = points.shape[0]
    i=0
    while i < N-1:
        k=2
        while i+k <N :
            if world.path_collisions(points[[i,i+k],:],margin).size == 0:
                ind.append(i+k-1)
                k = k +1
            else:
                break
        i += k - 1
    return np.delete(points, ind, 0)


def lin_interp(p, t):
    N = t.shape[0]

    c = np.zeros((N-1 , 2, 3))
    for i in range(N - 1):
        a=(p[i+1,:]-p[i,:])/(t[i+1]-t[i])
        b=p[i,:]-a*t[i]

        c[i,0,:] = a
        c[i,1,:] = b

    return c


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
        fig = plt.figure('Animation')
        ax = Axes3Ds(fig)
        world.draw(ax)
        ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
        ax.plot([goal[0]], [goal[1]], [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
        plt.show()

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.1, 0.1, 0.1])
        self.margin = 0.25

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        self.points = resample_points(world, self.path, self.margin)
        N = self.points.shape[0]
        dist = np.linalg.norm(np.diff(self.points, axis=0), axis=1)

        self.time = np.zeros((N,))
        for i, d in enumerate(dist):
            v = 0.8322 * np.log(d) + 0.9073
            if v < 0: v = 0.6
            # v=np.clip(v,0.2,0.8)
            self.time[i + 1] = self.time[i] + d / v


        self.coeff = min_jerk_zeros(self.points, self.time)
        # self.coeff = lin_interp(self.points, self.time)
        # self.times, self.ax, self.ay, self.az = cubic(self.points)

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
        x        = self.points[-1][:]
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0


        #### MINIMUM JERK
        if t <= self.time[-1]:
            i=sum(self.time<=t)-1
            x = self.coeff[i,0,:]*t**5+ self.coeff[i,1,:]*t**4+ self.coeff[i,2,:]*t**3+ self.coeff[i,3,:]*t**2+ self.coeff[i,4,:]*t+ self.coeff[i,5,:]
            x_dot = 5*self.coeff[i,0,:]*t**4 + 4*self.coeff[i,1,:]*t**3 + 3*self.coeff[i,2,:]*t**2 + 2*self.coeff[i,3,:]*t+ self.coeff[i,4,:]
            x_ddot = 20*self.coeff[i,0,:]*t**3 + 12*self.coeff[i,1,:]*t**2 + 6*self.coeff[i,2,:]*t+ 2*self.coeff[i,3,:]
            x_dddot = 60*self.coeff[i,0,:]*t**2 + 24*self.coeff[i,1,:]*t + 6*self.coeff[i,2,:]
            x_ddddot = 120*self.coeff[i,0,:]*t + 24*self.coeff[i,1,:]



        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output


