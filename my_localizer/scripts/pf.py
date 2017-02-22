#!/usr/bin/env python

""" This is the starter code for the robot localization project """

from __future__ import division

import rospy

from std_msgs.msg import Header, String, ColorRGBA
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Vector3, PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion
from nav_msgs.srv import GetMap
from visualization_msgs.msg import Marker, MarkerArray
from copy import deepcopy
import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
from random import gauss

import math
import time

import numpy as np
from scipy.stats import norm 
from numpy.random import random_sample
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from occupancy_field import OccupancyField

from helper_functions import (convert_pose_inverse_transform,
                              convert_translation_rotation_to_pose,
                              convert_pose_to_xy_and_theta,
                              angle_diff)

class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """
    def __init__(self,x=0.0,y=0.0,theta=0.0,w=1.0):
        """ Construct a new Particle
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized """ 
        self._w = w
        self.theta = theta
        self.x = x
        self.y = y

    @property
    def w(self): 
        return self._w


    @w.setter
    def w(self, value):
        # Do something if you want
        self._w = value

    @classmethod
    def from_numpy(self, numpy_array):
        numpy_array = numpy_array.flatten()[:]
        x = numpy_array[0]
        y = numpy_array[1]
        theta = numpy_array[2]
        w = numpy_array[3]
        return Particle(x, y, theta, w)

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        orientation_tuple = tf.transformations.quaternion_from_euler(0,0,self.theta)
        return Pose(position=Point(x=self.x,y=self.y,z=0), orientation=Quaternion(x=orientation_tuple[0], y=orientation_tuple[1], z=orientation_tuple[2], w=orientation_tuple[3]))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    # TODO: define additional helper functions if needed



class ParticleFilter:
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            initialized: a Boolean flag to communicate to other class methods that initializaiton is complete
            base_frame: the name of the robot base coordinate frame (should be "base_link" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            num_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            laser_max_distance: the maximum distance to an obstacle we should use in a likelihood calculation
            pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            laser_subscriber: listens for new scan data on topic self.scan_topic
            tf_listener: listener for coordinate transforms
            tf_broadcaster: broadcaster for coordinate transforms
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            map: the map we will be localizing ourselves in.  The map should be of type nav_msgs/OccupancyGrid
    """
    def __init__(self):
        self.initialized = False        # make sure we don't perform updates before everything is setup
        rospy.init_node('pf')           # tell roscore that we are creating a new node named "pf"

        self.base_frame = "base_link"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from 

        # If these are set any lower, the transform will timeout
        self.d_thresh = 0.1          # the amount of linear movement before performing an update
        self.a_thresh = math.pi/12       # the amount of angular movement before performing an update

        self.laser_max_distance = 2.0   # maximum penalty to assess in the likelihood field model

        self.num_particles = 300
        self.particle_movement_noise = 0.1
        self.sensor_variance = 0.05


        # Setup pubs and subs

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.pose_listener = rospy.Subscriber("initialpose", PoseWithCovarianceStamped, self.update_initial_pose)
        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = rospy.Publisher("particlecloud", MarkerArray, queue_size=10)

        # Publishes points (places where robot is guessing a wall is)
        self.point_publisher = rospy.Publisher('/visualization_messages/Marker', Marker, queue_size=10)

        # laser_subscriber listens for data from the lidar
        self.laser_subscriber = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_received)

        # enable listening for and broadcasting coordinate transforms
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

        self.particle_cloud = []

        self.current_odom_xy_theta = []

        # request the map from the map server, the map should be of type nav_msgs/OccupancyGrid
        occ_map = self.get_map_from_server()

        self.occupancy_field = OccupancyField(occ_map)
        self.initialized = True

        # Initialize at 0,0
        self.robot_pose = Pose()


    def get_map_from_server(self):
        print 'Waiting for Service at: ' + str(rospy.Time.now())
        rospy.wait_for_service('static_map')
        print 'Recieved Service at: ' + str(rospy.Time.now())
        try:
            get_map = rospy.ServiceProxy('static_map', GetMap)
            return get_map().map
        except rospy.ServiceException as e:
            print "Service call failed: %s"%e


    def update_robot_pose(self, particles):
        """ Update the estimate of the robot's pose given the updated particles.
            Compute the most likely pose (i.e. the mode of the distribution)
        """
        # first make sure that the particle weights are normalized
        particles = normalize_particles(particles)

        # Get the most likely particle
        if particles:
            weights = np.array([p.w for p in particles])
            max_index = np.argmax(weights)
            most_likely_pose = particles[max_index].as_pose()

            self.robot_pose = most_likely_pose

    def update_particles_with_odom(self, noise):
        """ Update the particles using the newly given odometry pose.
            The function computes the value delta which is a tuple (x,y,theta)
            that indicates the change in position and angle between the odometry
            when the particles were last updated and the current odometry.

            noise: a number between 0 and 1 which describes how much noise we will
            add to the movements
        """
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     new_odom_xy_theta[2] - self.current_odom_xy_theta[2])

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return

        d_x, d_y, d_theta = delta

        # move each particle the same distance in their coordinate frame
        new_particles = []
        for p in self.particle_cloud:
            angle_between_frames = old_odom_xy_theta[2] - p.theta
            res = rotate(angle_between_frames, d_x, d_y)
            new_d_x = res[0]
            new_d_y = res[1]
            rands = random_sample(3)
            pos = np.array([p.x + new_d_x, p.y + new_d_y, p.theta + d_theta])
            pos_noisy = pos*(1 + noise*(rands - 0.5))
            new_particle = Particle.from_numpy(np.append(pos_noisy, p.w)) 
            new_particles.append(new_particle)
        self.particle_cloud = new_particles

    def resample_particles(self, particles):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample.
        """
        print 'Resampling Particles ...'

        # make sure the distribution is normalized
        particles = normalize_particles(particles)
        choices = particles
        probabilities = np.array([p.w for p in particles])
        num_to_resample = len(particles)
        new_particles = draw_random_sample(choices, probabilities, num_to_resample)

        #TODO: DO we get new particles?
        return new_particles 

    def update_particles_with_laser(self, msg, particles, variance = 0.1):
        """ Updates the particle weights in response to the scan contained in the msg """
        print 'Updating Particle Weights with Laser'
        laser_angles = np.linspace(0, 360, 90).astype(int)
        rs = np.array(msg.ranges)[laser_angles]
        good_rs = rs[rs != 0]
        good_angles = laser_angles[rs != 0]
        phi = np.radians(good_angles)
        thetas = np.array([p.theta for p in self.particle_cloud])
        # Sum of combinations of angles. shape (size(phi), size(thetas))
        added_angles = thetas + phi[:, np.newaxis]
 

        # List of points that we will publish
        points = []
        for i, p in enumerate(particles):
            if np.any(good_rs):
                # Calculate the places where the point would be
                d = good_rs
                p_angs = added_angles[:,i]

                x = d*np.cos(p_angs)
                y = d*np.sin(p_angs)
                #import pdb
                #pdb.set_trace()

                # Publish the points!
                # for i in range(x.size):
                #     points.append(Point(p.x + x[i], p.y + y[i], 0))

                # How far away is that from a point on the map?
                dist = self.occupancy_field.get_closest_obstacle_distance_vectorized(p.x + x, p.y + y)
                # if there are nans
                if np.any(np.isnan(dist)):
                    # Set the weight to zero
                    print 'Erasing Nans...'
                    weight = 0
                else:
                    # Modeling distance as a normal distribution centerd around 0
                    closeness = norm.pdf(dist, variance)
                    # Add the closenesses together to figure out our weight
                    weight = np.sum((closeness**3)/closeness.size)
                    # print 'Total Match: ' + str(weight)
                p.w = weight
            else:
                p.w = 0.00001
                print "ERROR: NO POINTS!!!"


        self.publish_markers(points)

        return particles

    def publish_markers(self, points):
        """
        Helper functioning for publishing points.

        Args:
            points: points to be published
        """
        marker = Marker(
            type=Marker.POINTS,
            header=Header(
                stamp=rospy.Time.now(),
                frame_id=self.map_frame
            ),
            points=points,
            scale=Vector3(0.1, 0.1, 0.1),
            color=ColorRGBA(1.0, 1.0, 0.0, 1.0)
        )
        self.point_publisher.publish(marker)

    def create_arrow(self, number, pose, mag, color):
        marker = Marker(
            type=Marker.ARROW,
            id = number,
            header=Header(
                stamp=rospy.Time.now(),
                frame_id=self.map_frame
            ),
            pose=pose,
            scale=Vector3(100000 * (mag)**3 + 0.1, mag + 0.05, mag + 0.05),
            color=ColorRGBA(color[0], color[1], color[2], color[3])
        )
        return marker

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        xy_theta = convert_pose_to_xy_and_theta(msg.pose.pose)
        self.particle_cloud = self.initialize_particle_cloud(xy_theta)
        self.fix_map_to_odom_transform(msg)

    def initialize_particle_cloud(self, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is ommitted, the odometry will be used """
        print 'Initializing Particle Cloud...'
        if xy_theta == None:
            xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)

        xs, ys, thetas = gen_random_particle_positions(xy_theta, (0.25, 0.25, math.pi/7), self.num_particles)


        particle_cloud = [Particle(x, y, theta, 1) for (x, y, theta) in zip(xs, ys, thetas)]

        normalized_particle_cloud  = normalize_particles(particle_cloud)
        self.update_robot_pose(normalized_particle_cloud)
        return normalized_particle_cloud

    def publish_particles(self, msg):
        markers = []
        cmap_name = 'jet_r'
        alpha = 0.5
        weights = np.array([p.w for p in self.particle_cloud])
        # Generate a colormap for the weights
        cm = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax = np.max(weights))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        vals = scalarMap.to_rgba(weights, alpha=0.5)

        for i, p in enumerate(self.particle_cloud):
            # The the color of a given particle
            particle_color = vals[i, :]
            marker = self.create_arrow(i, p.as_pose(), p.w, particle_color)
            markers.append(marker)
        #nprint_weights(self.particle_cloud)

        # actually send the message so that we can view it in rviz
        self.particle_pub.publish(MarkerArray(markers=markers))

    def scan_received(self, msg):
        """ This is the default logic for what to do when processing scan data.
            Feel free to modify this, however, I hope it will provide a good
            guide.  The input msg is an object of type sensor_msgs/LaserScan """
        if not(self.initialized):
            # wait for initialization to complete
            return

        if not(self.tf_listener.canTransform(self.base_frame,msg.header.frame_id,msg.header.stamp)):
            # need to know how to transform the laser to the base frame
            # this will be given by either Gazebo or neato_node
            return

        if not(self.tf_listener.canTransform(self.base_frame,self.odom_frame,msg.header.stamp)):
            # need to know how to transform between base and odometric frames
            # this will eventually be published by either Gazebo or neato_node
            return

        # calculate pose of laser relative ot the robot base
        p = PoseStamped(header=Header(stamp=rospy.Time(0),
                                      frame_id=msg.header.frame_id))
        self.laser_pose = self.tf_listener.transformPose(self.base_frame,p)

        # find out where the robot thinks it is based on its odometry
        p = PoseStamped(header=Header(stamp=msg.header.stamp,
                                      frame_id=self.base_frame),
                        pose=Pose())
        self.odom_pose = self.tf_listener.transformPose(self.odom_frame, p)
        # store the the odometry pose in a more convenient format (x,y,theta)
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)

        if not(self.particle_cloud):
            # print 'no particle cloud'
            # now that we have all of the necessary transforms we can update the particle cloud
            self.particle_cloud = self.initialize_particle_cloud()
            # cache the last odometric pose so we can only update our particle filter if we move more than self.d_thresh or self.a_thresh
            self.current_odom_xy_theta = new_odom_xy_theta
            # update our map to odom transform now that the particles are initialized
            self.fix_map_to_odom_transform(msg)
        elif (math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or
              math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or
              math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh):
            # we have moved far enough to do an update!
            self.update_particles_with_odom(noise=self.particle_movement_noise)    # update based on odometry
            self.particle_cloud = self.update_particles_with_laser(msg, self.particle_cloud, self.sensor_variance)   # update based on laser scan
            self.update_robot_pose(self.particle_cloud)                # update robot's pose
            self.particle_cloud = self.resample_particles(self.particle_cloud)               # resample particles to focus on areas of high density
            self.fix_map_to_odom_transform(msg)     # update map to odom transform now that we have new particles
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg)

    def fix_map_to_odom_transform(self, msg):
        """ This method constantly updates the offset of the map and 
            odometry coordinate systems based on the latest results from
            the localizer
            TODO: if you want to learn a lot about tf, reimplement this... I can provide
                  you with some hints as to what is going on here. """
        (translation, rotation) = convert_pose_inverse_transform(self.robot_pose)
        p = PoseStamped(pose=convert_translation_rotation_to_pose(translation,rotation),
                        header=Header(stamp=msg.header.stamp,frame_id=self.base_frame))
        self.tf_listener.waitForTransform(self.base_frame, self.odom_frame, msg.header.stamp, rospy.Duration(1.0))
        self.odom_to_map = self.tf_listener.transformPose(self.odom_frame, p)
        (self.translation, self.rotation) = convert_pose_inverse_transform(self.odom_to_map.pose)

    def broadcast_last_transform(self):
        """ Make sure that we are always broadcasting the last map
            to odom transformation.  This is necessary so things like
            move_base can work properly. """
        if not(hasattr(self,'translation') and hasattr(self,'rotation')):
            return
        self.tf_broadcaster.sendTransform(self.translation,
                                          self.rotation,
                                          rospy.get_rostime(),
                                          self.odom_frame,
                                          self.map_frame)

def rotate(angle, x, y):
    rot = np.matrix([[np.cos(angle), -1*np.sin(angle)],
                [np.sin(angle) ,    np.cos(angle)]])
    destination = np.matrix([x, y]) * rot
    return np.array(destination).flatten()

def gen_random_particle_positions(xy_theta, std_devs, size):
    """generates random particles positions centered at xy_theta with std deviations defined by std_devs"""
    x, y, theta = xy_theta
    x_dev, y_dev, theta_dev = std_devs
    xs = np.random.normal(x, x_dev, size)
    ys = np.random.normal(y, y_dev, size)
    thetas = np.random.normal(theta, theta_dev, size)
    return (xs, ys, thetas)

def normalize_particles(particles):
    """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """
    if particles:
        weights = np.array([p.w for p in particles])
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            normed_weights = weights / sum_weights
            for (p, weight) in zip(particles, normed_weights):
                p.w = weight
    return particles


def draw_random_sample(choices, probabilities, n):
    """ Return a random sample of n elements from the set choices with the specified probabilities
        choices: the values to sample from represented as a list
        probabilities: the probability of selecting each element in choices represented as a list
        n: the number of samples
    """
    values = np.array(range(len(choices)))
    probs = np.array(probabilities)
    bins = np.add.accumulate(probs)
    try:
        inds = values[np.digitize(random_sample(n), bins)]
    except IndexError as e:
        import pdb
        pdb.set_trace()
    samples = []
    for i in inds:
        samples.append(deepcopy(choices[int(i)]))
    return samples

def print_weights(particle_cloud):
    """ 
    Print out all the weights of the particles in the particle_cloud
    particle_cloud: list of particles to print out weights of
    """
    weights = ' '
    for p in n.particle_cloud:
        weights += str(p.w)
        weights += ' '
    print weights 

if __name__ == '__main__':
    n = ParticleFilter()
    r = rospy.Rate(5)

    while not(rospy.is_shutdown()):
        # in the main loop all we do is continuously broadcast the latest map to odom transform
        n.broadcast_last_transform()
        r.sleep()