import inspect
import json
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import time

from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World

from occupancy_map_lab import OccupancyMap
from se3_control_lab import SE3Control
from world_traj_lab import WorldTraj

import rosbag
from scipy.spatial.transform import Rotation as R

# Improve figure display on high DPI screens.
# mpl.rcParams['figure.dpi'] = 200

# VICON DATA
bagfile = '../proj1_3/util/map_3_0.bag'
state_v = {}
control_v = {}
i_s = 165
i_f = 985
with rosbag.Bag(bagfile) as bag:
    odometry = np.array([
        np.array([t.to_sec() - bag.get_start_time(),
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z,
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
            for (_, msg, t) in bag.read_messages(topics=['odom'])])
    vicon_time = odometry[i_s:i_f, 0] - odometry[i_s, 0]
    state_v['x'] = odometry[i_s:i_f, 1:4]
    state_v['v'] = odometry[i_s:i_f, 4:7]
    state_v['w'] = odometry[i_s:i_f, 7:10]
    state_v['q'] = odometry[i_s:i_f, 10:15]
    commands = np.array([
        np.array([t.to_sec() - bag.get_start_time(),
        msg.linear.z, msg.linear.y, msg.linear.x])
            for (_, msg, t) in bag.read_messages(topics=['so3cmd_to_crazyflie/cmd_vel_fast'])])
    cmd = commands[i_s:i_f,:]
    command_time = cmd[:, 0]-cmd[0, 0]
    c1 = -0.6709 # Coefficients to convert thrust PWM to Newtons.
    c2 = 0.1932
    c3 = 13.0652
    control_v['cmd_thrust'] = (((cmd[:, 1]/60000 - c1) / c2)**2 - c3)/1000*9.81
    control_v['cmd_q'] = R.from_euler('zyx', np.transpose([cmd[:, 2], cmd[:, 3],
                                                         np.zeros(cmd[:, 2].shape)]), degrees=True).as_quat()

# Choose a test example file. You should write your own example files too!
# filename = '../proj1_3/util/test_lab_1.json'
# filename = '../proj1_3/util/test_lab_2.json'
filename = '../proj1_3/util/test_lab_3.json'
# filename = '../util/test_window.json'
# filename = '../util/test_maze.json'
# filename = '../util/test_over_under.json'
# filename = '../util/test_nik.json'

# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

# This object defines the quadrotor dynamical model and should not be changed.
quadrotor = Quadrotor(quad_params)
robot_radius = 0.25

# Your SE3Control object (from project 1-1).
my_se3_control = SE3Control(quad_params)

# Your MapTraj object. This behaves like the trajectory function you wrote in
# project 1-1, except instead of giving it waypoints you give it the world,
# start, and goal.
planning_start_time = time.time()
my_world_traj = WorldTraj(world, start, goal)
planning_end_time = time.time()

# Help debug issues you may encounter with your choice of resolution and margin
# by plotting the occupancy grid after inflation by margin. THIS IS VERY SLOW!!
# fig = plt.figure('world')
# ax = Axes3Ds(fig)
# world.draw(ax)
# fig = plt.figure('occupancy grid')
# ax = Axes3Ds(fig)
# resolution = SET YOUR RESOLUTION HERE
# margin = SET YOUR MARGIN HERE
# oc = OccupancyMap(world, resolution, margin)
# oc.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
# plt.show()

# Set simulation parameters.
t_final = 60
initial_state = {'x': start,
                 'v': (0, 0, 0),
                 'q': (0, 0, 0, 1), # [i,j,k,w]
                 'w': (0, 0, 0)}

# Perform simulation.
#
# This function performs the numerical simulation.  It returns arrays reporting
# the quadrotor state, the control outputs calculated by your controller, and
# the flat outputs calculated by you trajectory.

print()
print('Simulate.')
(sim_time, state, control, flat, exit) = simulate(initial_state,
                                              quadrotor,
                                              my_se3_control,
                                              my_world_traj,
                                              t_final)
print(exit.value)

# Print results.
#
# Only goal reached, collision test, and flight time are used for grading.

collision_pts = world.path_collisions(state['x'], robot_radius)

stopped_at_goal = (exit == ExitStatus.COMPLETE) and np.linalg.norm(state['x'][-1] - goal) <= 0.05
no_collision = collision_pts.size == 0
flight_time = sim_time[-1]
flight_distance = np.sum(np.linalg.norm(np.diff(state['x'], axis=0),axis=1))
planning_time = planning_end_time - planning_start_time

print()
print(f"Results:")
print(f"  No Collision:    {'pass' if no_collision else 'FAIL'}")
print(f"  Stopped at Goal: {'pass' if stopped_at_goal else 'FAIL'}")
print(f"  Flight time:     {flight_time:.1f} seconds")
print(f"  Flight distance: {flight_distance:.1f} meters")
print(f"  Planning time:   {planning_time:.1f} seconds")
if not no_collision:
    print()
    print(f"  The robot collided at location {collision_pts[0]}!")

# Plot Results
#
# You will need to make plots to debug your quadrotor.
# Here are some example of plots that may be useful.

# # Visualize the original dense path from A*, your sparse waypoints, and the
# # smooth trajectory.
# fig = plt.figure('A* Path, Waypoints, and Trajectory')
# ax = Axes3Ds(fig)
# world.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
# if hasattr(my_world_traj, 'path'):
#     if my_world_traj.path is not None:
#         world.draw_line(ax, my_world_traj.path, color='red', linewidth=1)
# else:
#     print("Have you set \'self.path\' in WorldTraj.__init__?")
# if hasattr(my_world_traj, 'points'):
#     if my_world_traj.points is not None:
#         world.draw_points(ax, my_world_traj.points, color='purple', markersize=8)
# else:
#     print("Have you set \'self.points\' in WorldTraj.__init__?")
# world.draw_line(ax, flat['x'], color='black', linewidth=2)
# ax.legend(handles=[
#     Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
#     Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
#     Line2D([], [], color='black', linewidth=2, label='Trajectory')],
#     loc='upper right')
#
# # Position and Velocity vs. Time
# (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time')
# x = state['x']
# x_des = flat['x']
# ax = axes[0]
# ax.plot(sim_time, x_des[:,0], 'r', sim_time, x_des[:,1], 'g', sim_time, x_des[:,2], 'b')
# ax.plot(sim_time, x[:,0], 'r.',    sim_time, x[:,1], 'g.',    sim_time, x[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('position, m')
# ax.grid('major')
# ax.set_title('Position')
# v = state['v']
# v_des = flat['x_dot']
# ax = axes[1]
# ax.plot(sim_time, v_des[:,0], 'r', sim_time, v_des[:,1], 'g', sim_time, v_des[:,2], 'b')
# ax.plot(sim_time, v[:,0], 'r.',    sim_time, v[:,1], 'g.',    sim_time, v[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('velocity, m/s')
# ax.set_xlabel('time, s')
# ax.grid('major')
#
# # Orientation and Angular Velocity vs. Time
# (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time')
# q_des = control['cmd_q']
# q = state['q']
# ax = axes[0]
# ax.plot(sim_time, q_des[:,0], 'r', sim_time, q_des[:,1], 'g', sim_time, q_des[:,2], 'b', sim_time, q_des[:,3], 'k')
# ax.plot(sim_time, q[:,0], 'r.',    sim_time, q[:,1], 'g.',    sim_time, q[:,2], 'b.',    sim_time, q[:,3],     'k.')
# ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
# ax.set_ylabel('quaternion')
# ax.set_xlabel('time, s')
# ax.grid('major')
# w = state['w']
# ax = axes[1]
# ax.plot(sim_time, w[:,0], 'r.', sim_time, w[:,1], 'g.', sim_time, w[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('angular velocity, rad/s')
# ax.set_xlabel('time, s')
# ax.grid('major')
#
# # Commands vs. Time
# (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
# s = control['cmd_motor_speeds']
# ax = axes[0]
# ax.plot(sim_time, s[:,0], 'r.', sim_time, s[:,1], 'g.', sim_time, s[:,2], 'b.', sim_time, s[:,3], 'k.')
# ax.legend(('1', '2', '3', '4'), loc='upper right')
# ax.set_ylabel('motor speeds, rad/s')
# ax.grid('major')
# ax.set_title('Commands')
# M = control['cmd_moment']
# ax = axes[1]
# ax.plot(sim_time, M[:,0], 'r.', sim_time, M[:,1], 'g.', sim_time, M[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('moment, N*m')
# ax.grid('major')
# T = control['cmd_thrust']
# ax = axes[2]
# ax.plot(sim_time, T, 'k.')
# ax.set_ylabel('thrust, N')
# ax.set_xlabel('time, s')
# ax.grid('major')
#
# # 3D Paths
# fig = plt.figure('3D Path')
# ax = Axes3Ds(fig)
# world.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
# world.draw_line(ax, flat['x'], color='black', linewidth=2)
# world.draw_points(ax, state['x'], color='blue', markersize=4)
# if collision_pts.size > 0:
#     ax.plot(collision_pts[0,[0]], collision_pts[0,[1]], collision_pts[0,[2]], 'rx', markersize=36, markeredgewidth=4)
# ax.legend(handles=[
#     Line2D([], [], color='black', linewidth=2, label='Trajectory'),
#     Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
#     loc='upper right')
#
#
# # Animation (Slow)
# #
# # Instead of viewing the animation live, you may provide a .mp4 filename to save.
#
# R = Rotation.from_quat(state['q']).as_dcm()
# animate(sim_time, state['x'], R, world=world, filename=None, show_axes=True)
#
# plt.show()

# -----------------------------------------------------------------------------
#                      Simulation Vs. LAB
# -----------------------------------------------------------------------------

# A* planned Vs. Simulation Vs. Vicon observed
fig = plt.figure('A* Path, Waypoints, and Trajectory')
ax = Axes3Ds(fig)
world.draw(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
world.draw_line(ax, my_world_traj.path, color='red', linewidth=1)
world.draw_points(ax, my_world_traj.points, color='purple', markersize=8)
world.draw_line(ax, state_v['x'], color='black', linewidth=2)
world.draw_line(ax, flat['x'], color='green', linewidth=2)
ax.legend(handles=[
    Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
    Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='A* Sparse Waypoints'),
    Line2D([], [], color='green', linewidth=1, label='Simulation Trajectory'),
    Line2D([], [], color='black', linewidth=2, label='CrazyFlie Trajectory (VICON)')],loc='upper right')

# Position vs. Time
(fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Position vs Time')
x = state['x']
x_des = flat['x']
ax = axes[0]
ax.plot(sim_time, x[:,0], 'r')
ax.plot(sim_time, x_des[:,0], 'b')
ax.plot(vicon_time, state_v['x'][:,0], 'k')
ax.legend(('Planned', 'Simulation', 'LAB'), loc='lower right')
ax.set_ylabel('X, m')
ax.grid('major')
ax.set_xlabel('Time, s')
ax = axes[1]
ax.plot(sim_time, x[:,1], 'r')
ax.plot(sim_time, x_des[:,1], 'b')
ax.plot(vicon_time, state_v['x'][:,1], 'k')
ax.legend(('Planned', 'Simulation', 'LAB'), loc='lower right')
ax.set_ylabel('Y, m')
ax.grid('major')
ax.set_xlabel('Time, s')
ax = axes[2]
ax.plot(sim_time, x[:,2], 'r')
ax.plot(sim_time, x_des[:,2], 'b')
ax.plot(vicon_time, state_v['x'][:,2], 'k')
ax.legend(('Planned', 'Simulation', 'LAB'), loc='lower right')
ax.set_ylabel('Z, m')
ax.grid('major')
ax.set_xlabel('Time, s')

(fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Velocity vs Time')
v = state['v']
v_des = flat['x_dot']
ax = axes[0]
ax.plot(sim_time, v[:,0], 'r')
ax.plot(sim_time, v_des[:,0], 'b')
ax.plot(vicon_time, state_v['v'][:,0], 'k')
ax.legend(('Planned', 'Simulation', 'LAB'), loc='lower right')
ax.set_ylabel('V_x, m/sec')
ax.grid('major')
ax.set_xlabel('Time, s')
ax = axes[1]
ax.plot(sim_time, v[:,1], 'r')
ax.plot(sim_time, v_des[:,1], 'b')
ax.plot(vicon_time, state_v['v'][:,1], 'k')
ax.legend(('Planned', 'Simulation', 'LAB'), loc='lower right')
ax.set_ylabel('V_y, m/sec')
ax.grid('major')
ax.set_xlabel('Time, s')
ax = axes[2]
ax.plot(sim_time, v[:,2], 'r')
ax.plot(sim_time, v_des[:,2], 'b')
ax.plot(vicon_time, state_v['v'][:,2], 'k')
ax.legend(('Planned', 'Simulation', 'LAB'), loc='lower right')
ax.set_ylabel('V_z, m/sec')
ax.grid('major')
ax.set_xlabel('Time, s')


(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time')
q_des = control_v['cmd_q']
q = state_v['q']
ax = axes[1]
ax.plot(command_time, q_des[:, 0], 'r', command_time, q_des[:, 1], 'g',
        command_time, q_des[:, 2], 'b', command_time, q_des[:, 3], 'k')
ax.plot(vicon_time, q[:, 0], 'r.',    vicon_time, q[:, 1], 'g.',
        vicon_time, q[:, 2], 'b.',    vicon_time, q[:, 3],     'k.')
ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
ax.set_ylabel('quaternion')
ax.set_xlabel('time, s')
ax.grid('major')
ax.set_title('CrazyFlie')
q_des = control['cmd_q']
q = state['q']
ax = axes[0]
ax.plot(sim_time, q_des[:, 0], 'r', sim_time, q_des[:, 1], 'g',
        sim_time, q_des[:, 2], 'b', sim_time, q_des[:, 3], 'k')
ax.plot(sim_time, q[:, 0], 'r.',    sim_time, q[:, 1], 'g.',
        sim_time, q[:, 2], 'b.',    sim_time, q[:, 3],     'k.')
ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
ax.set_ylabel('quaternion')
ax.set_xlabel('time, s')
ax.grid('major')
ax.set_title('Simulation')


(fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Angular Velocity vs Time')
w = state['w']
ax = axes[0]
ax.plot(sim_time, w[:,0], 'r')
ax.plot(vicon_time, state_v['w'][:,0], 'k')
ax.legend(('Simulation', 'LAB'), loc='upper right')
ax.grid('major')
ax.set_ylabel('w_x, rad/s')
ax.set_xlabel('Time, s')
ax = axes[1]
ax.plot(sim_time, w[:,1], 'r')
ax.plot(vicon_time, state_v['w'][:,1], 'k')
ax.legend(('Simulation', 'LAB'), loc='upper right')
ax.set_ylabel('w_y, rad/s')
ax.set_xlabel('Time, s')
ax.grid('major')
ax = axes[2]
ax.plot(sim_time, w[:,2], 'r')
ax.plot(vicon_time, state_v['w'][:,2], 'k')
ax.legend(('Simulation', 'LAB'), loc='upper right')
ax.grid('major')
ax.set_ylabel('w_z, rad/s')
ax.set_xlabel('Time, s')


fig = plt.figure('Thrust vs Time')
T = control['cmd_thrust']
plt.plot(sim_time, T, 'r')
plt.plot(command_time, control_v['cmd_thrust'], 'k')
plt.legend(('Simulation', 'LAB'), loc='lower right')
plt.ylabel('thrust, N')
plt.xlabel('time, s')
plt.grid('major')
plt.title('Commanded Thrust')

plt.show()
