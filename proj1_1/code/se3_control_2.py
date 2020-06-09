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


        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2


        k = self.k_drag/self.k_thrust
        L = self.arm_length
        self.u2f = np.array([[1,  1,  1,  1],
                             [ 0,  L,  0, -L],
                             [-L,  0,  L,  0],
                             [ k, -k,  k, -k]])
        self.inv = np.linalg.inv(self.u2f)
        self.down = np.array([0, 0, self.mass*self.g])

        # self.Kp = np.diag(np.array([500,500,500]))
        # self.Kd = np.diag(np.array([100,100,100]))
        #
        # self.K_p = np.diag(np.array([9,9,9]))
        # self.K_d = np.diag(np.array([10,10,10]))
        self.Kp = np.diag(np.array([280,280,250]))
        self.Kd = np.diag(np.array([70,70,60]))

        self.K_p = np.diag(np.array([6,6,6]))
        self.K_d = np.diag(np.array([5,5,5]))



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
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))


        error_v = state['v']-flat_output['x_dot']
        error_x = state['x']-flat_output['x']


        r_ddot_des = flat_output['x_ddot'] - self.K_d .dot(error_v) - self.K_p .dot(error_x)

        # phi_des=(r_ddot_des[0]-r_ddot_des[1])/(self.g*(np.cos(flat_output['yaw'])+np.sin(flat_output['yaw'])))
        # theta_des=(1/self.g)*r_ddot_des[0]-phi_des*np.sin(flat_output['yaw'])

        phi_des=(r_ddot_des[0]*np.sin(flat_output['yaw'])-r_ddot_des[1]*np.cos(flat_output['yaw']))/self.g
        theta_des=r_ddot_des[0]/(self.g*np.cos(flat_output['yaw']))-phi_des*np.tan(flat_output['yaw'])

        u1=self.mass*( self.g + r_ddot_des[2] )

        r=Rotation.from_quat(state['q']).as_euler('zxy', degrees=False)

        e_p=state['w'][0]-0
        e_q=state['w'][1]-0
        e_r=state['w'][2]-flat_output['yaw_dot']

        e_phi=r[1]-phi_des
        e_theta=r[2]-theta_des
        e_y=r[0]-flat_output['yaw']

        u2=self.inertia @ (-self.Kp.dot(np.array([e_phi,e_theta,e_y]))- self.Kd.dot( np.array([e_p,e_q,e_r])))

        u = np.array([u1, u2[0], u2[1], u2[2]])

        r_sp = np.sqrt(np.clip((self.inv.dot(u)) / self.k_thrust ,0, a_max=None))

        cmd_motor_speeds = np.clip(r_sp, self.rotor_speed_min, self.rotor_speed_max)
        cmd_thrust = np.clip(u1, 0, a_max=None)
        cmd_moment = u2

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}

        return control_input
