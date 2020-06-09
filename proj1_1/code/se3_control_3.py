import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2


        # STUDENT CODE HERE

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """

        "First extract the current (actual) position of the robot and the desired position derived from the waypoints"
        ri = state["x"]             #expressed in inertial frame
        rT = flat_output["x"]       #expressed in inertial frame
        "actual velocity of robot and desired velocity from waypoints"
        vi = state["v"]             #expressed in inertial frame
        vT = flat_output["x_dot"]    #expressed in inertial frame
        "Create the position and velocity error vector between those two"
        e_pos = ri - rT
        e_vel = vi - vT
        rdotdotT = flat_output["x_ddot"]   #acceleration as returned from waypoints
        psiT = flat_output["yaw"]

        "From quaternions to euler angles"
        rot = Rotation.from_quat(state["q"])
        eulerAngles = rot.as_euler('zyx')
        psi = eulerAngles[0]
        theta = eulerAngles[1]
        phi = eulerAngles[2]

        kd = 5                      #tune
        kp = 5                      #tune

        kdMatrix = np.diagflat(kd*np.ones((1,3)))
        kpMatrix = np.diagflat(kp*np.ones((1,3)))

        rdotdotdes = rdotdotT - kdMatrix@e_vel - kpMatrix@e_pos

        "Now we can compute u1 (thrust), θDes and φDes"
        u1 = self.mass*(self.g + rdotdotdes[2])      #thrust control
        phiDes = rdotdotdes[0]*np.sin(psiT) - rdotdotdes[1]*np.cos(psiT)
        thetaDes = (rdotdotdes[0] - phiDes*np.sin(psiT)) / self.g*np.cos(psiT)

        "Now we can compute u2"
        I = np.diagflat([self.Ixx, self.Iyy, self.Izz])
        u2 = I@np.array([[-kp*(phi - phiDes)-kd*(state["w"][0])],
                         [-kp*(theta - thetaDes)-kd*(state["w"][1])],
                         [-kp*(psi - psiT)-kd*(state["w"][2])]])

        "From u2 to forces"
        gamma = self.k_drag / self.k_thrust
        A = np.array([[1, 1, 1, 1],
                      [0, self.arm_length, 0, -self.arm_length],
                      [-self.arm_length, 0, self.arm_length, 0],
                      [gamma, -gamma, gamma, -gamma]])

        forces = np.linalg.inv(A)@np.insert(u2,0,u1)

        "From forces to motor speeds"
        cmd_motor_speeds = np.squeeze(np.sqrt(1/self.k_thrust * forces))
        cmd_thrust = np.sum(forces)

        "From motor speeds to moments"
        cmd_moment = u2
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
