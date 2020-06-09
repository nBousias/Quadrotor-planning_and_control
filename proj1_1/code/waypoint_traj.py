import numpy as np

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
        # Find the overall distance needed to travel
        dist = np.zeros((len(points) - 1,))

        for i in range(len(points) - 1):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[i + 1]
            x_dist = np.sqrt(np.abs(x2 ** 2 - x1 ** 2))
            y_dist = np.sqrt(np.abs(y2 ** 2 - y1 ** 2))
            z_dist = np.sqrt(np.abs(z2 ** 2 - z1 ** 2))
            dist[i] = x_dist + y_dist + z_dist

        # Find the time needed to be at each point
        #   - Average speed is determined based on the length of the leg needed to travel
        time = np.zeros((len(dist), ))
        for i, d in enumerate(dist):
            if d < 1.5:
                time[i] = dist[i] / 0.9
            elif d < 4:
                time[i] = dist[i] / 1.2
            else:
                time[i] = dist[i] / 2.6

        times = [0]
        for i in range(len(time)):
            times.append(times[i] + time[i])

        for i, t in enumerate(times):
            if i == 1:
                times[i] = t + 0.1
            elif i == 0 or i == len(times) - 1:
                times[i] = t
            else:
                times[i] = t - 0.2 / (len(times))

        # Find the coefficients for the equations of motion
        ax = np.zeros((len(points) - 1, 4))
        ay = np.zeros((len(points) - 1, 4))
        az = np.zeros((len(points) - 1, 4))

        for i in range(len(points) - 1):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[i + 1]

            if i == 0 and len(points) > 2:
                vx1, vy1, vz1 = np.zeros((3,))
                vx2, vy2, vz2 = (points[i + 2] - points[i]) / np.linalg.norm(points[i + 2] - points[i])
            elif i == len(points) - 2 and len(points) > 2:
                vx1, vy1, vz1 = (points[i + 1] - points[i-1]) / np.linalg.norm(points[i + 1] - points[i-1])
                vx2, vy2, vz2 = np.zeros((3,))
            elif len(points) == 2 or i == len(points) - 1:
                vx1, vy1, vz1 = np.zeros((3,))
                vx2, vy2, vz2 = np.zeros((3,))
            else:
                vx1, vy1, vz1 = (points[i + 1] - points[i-1]) / np.linalg.norm(points[i + 1] - points[i-1])
                vx2, vy2, vz2 = (points[i + 2] - points[i]) / np.linalg.norm(points[i + 2] - points[i])

            t1 = times[i]
            t2 = times[i + 1]
            ax[i] = np.linalg.inv([[1, t1, t1 ** 2, t1 ** 3], [0, 1, 2 * t1, 3 * t1 ** 2], [1, t2, t2 ** 2, t2 ** 3],
                                   [0, 1, 2 * t2, 3 * t2 ** 2]]) @ np.transpose([x1, vx1, x2, vx2])
            ay[i] = np.linalg.inv([[1, t1, t1 ** 2, t1 ** 3], [0, 1, 2 * t1, 3 * t1 ** 2], [1, t2, t2 ** 2, t2 ** 3],
                                   [0, 1, 2 * t2, 3 * t2 ** 2]]) @ np.transpose([y1, vy1, y2, vy2])
            az[i] = np.linalg.inv([[1, t1, t1 ** 2, t1 ** 3], [0, 1, 2 * t1, 3 * t1 ** 2], [1, t2, t2 ** 2, t2 ** 3],
                                   [0, 1, 2 * t2, 3 * t2 ** 2]]) @ np.transpose([z1, vz1, z2, vz2])

        self.points = points
        self.times = times
        self.ax = ax
        self.ay = ay
        self.az = az


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
        for i in range(len(self.times) - 1):
            if t > self.times[i] and t < self.times[i + 1]:
                # Get the coefficients
                x0, x1, x2, x3 = self.ax[i]
                y0, y1, y2, y3 = self.ay[i]
                z0, z1, z2, z3 = self.az[i]
                # Find the equations of motions and their derivatives
                x = np.array([x0 + x1 * t + x2 * t ** 2 + x3 * t ** 3, y0 + y1 * t + y2 * t ** 2 + y3 * t ** 3,
                              z0 + z1 * t + z2 * t ** 2 + z3 * t ** 3])
                x_dot = np.array([x1 + 2 * x2 * t + 3 * x3 * t ** 2, y1 + 2 * y2 * t + 3 * y3 * t ** 2,
                                  z1 + 2 * z2 * t + 3 * z3 * t ** 2])
                x_ddot = np.array([2 * x2 + 6 * x3 * t, 2 * y2 + 6 * y3 * t, 2 * z2 + 6 * z3 * t])
                x_dddot = np.array([6 * x3, 6 * y3, 6 * z3])
                break
            elif self.times[i] == t:
                x, y, z = self.points[i]
                x = np.array([x, y, z])
                break
            elif t >= self.times[-1]:
                x, y, z = self.points[-1]
                x = np.array([x, y, z])
                break

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output


# import numpy as np
#
# class WaypointTraj(object):
#     """
#
#     """
#     def __init__(self, points):
#         """
#         This is the constructor for the Trajectory object. A fresh trajectory
#         object will be constructed before each mission. For a waypoint
#         trajectory, the input argument is an array of 3D destination
#         coordinates. You are free to choose the times of arrival and the path
#         taken between the points in any way you like.
#
#         You should initialize parameters and pre-compute values such as
#         polynomial coefficients here.
#
#         Inputs:
#             points, (N, 3) array of N waypoint coordinates in 3D
#         """
#
#         N=points.shape[0]
#         dp=np.diff(points,axis=0)
#         v=2.2
#         t_0 = 0.05
#         self.k=2.2
#         if np.sum(dp > 9) > 0:
#             t_0 = 0.5
#             v=1.5
#             self.k = 1
#         dt=np.linalg.norm(dp,axis=1)/v
#         vel=dp/dt[:,None]
#         timer=0
#
#         t=[]
#         for i in dt:
#             timer+=i+t_0
#             t.append(timer)
#
#         self.N=N
#         self.dt=dt
#         self.t0=t_0
#         self.p=points
#         self.dp=dp
#         self.time=np.array(t)
#         self.vel=np.pad(vel, (0, 1), 'constant')[:,0:3]
#
#
#     def update(self, t):
#         """
#         Given the present time, return the desired flat output and derivatives.
#
#         Inputs
#             t, time, s
#         Outputs
#             flat_output, a dict describing the present desired flat outputs with keys
#                 x,        position, m
#                 x_dot,    velocity, m/s
#                 x_ddot,   acceleration, m/s**2
#                 x_dddot,  jerk, m/s**3
#                 x_ddddot, snap, m/s**4
#                 yaw,      yaw angle, rad
#                 yaw_dot,  yaw rate, rad/s
#         """
#         x        = np.zeros((3,))
#         x_dot    = np.zeros((3,))
#         x_ddot   = np.zeros((3,))
#         x_dddot  = np.zeros((3,))
#         x_ddddot = np.zeros((3,))
#         yaw = 0
#         yaw_dot = 0
#
#
#
#         i = np.sum(self.time <= t)
#
#         x = self.p[i, :]
#         x_dot = self.k*self.vel[i,:]
#
#
#         if i<self.N-1:
#             if self.dp[i,0]==0:
#                 yaw=np.sign(self.dp[i,1])*np.pi/2
#             else:
#                 yaw=np.arctan(self.dp[i,1]/self.dp[i,0])
#
#
#         flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
#                         'yaw':yaw, 'yaw_dot':yaw_dot}
#         return flat_output
#
#
#
