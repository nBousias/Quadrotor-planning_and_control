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


        self.Kd = np.diag(np.array([10, 10, 10]))
        self.Kp = np.diag(np.array([10, 10, 10]))/np.sqrt(2)
        self.Kr = np.diag(np.array([500, 500, 500]))
        self.Kw = np.diag(np.array([100, 100, 100]))


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

        F_des = self.mass * (flat_output['x_ddot'] - self.Kd@error_v - self.Kp @ error_x + np.array([0,0,self.g]))

        R = Rotation.from_quat(state['q']).as_matrix()

        b3 = R[:,2]

        u1 = b3.dot(F_des)

        b3_des = F_des/np.linalg.norm(F_des)

        a=np.array([ np.cos(flat_output['yaw']) , np.sin(flat_output['yaw']) , 0 ])

        b2_des=np.cross(b3_des,a)/np.linalg.norm(np.cross(b3_des,a))

        R_des=np.array([ np.cross(b2_des,b3_des) , b2_des , b3_des ]).T



        aux= R_des.T @ R - R.T @ R_des
        er = 0.5 * np.array([aux[2, 1],aux[0, 2] ,aux[1, 0]])

        ew=state['w']

        u2= self.inertia @ ( -(self.Kr @ er) - (self.Kw @ ew ))


        u=np.array([u1,u2[0],u2[1],u2[2]])

        aux2=(self.inv @ u)/self.k_thrust
        r_sp=np.sign(aux2)*np.sqrt(np.abs(aux2))

        cmd_motor_speeds=np.clip(r_sp,self.rotor_speed_min,self.rotor_speed_max)
        cmd_thrust=np.clip(u1,0,a_max=None)
        cmd_moment=u2

        cmd_q=Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
