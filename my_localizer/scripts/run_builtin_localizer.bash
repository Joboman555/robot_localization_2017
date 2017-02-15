#!/bin/bash

commands () {
    # pgrep roscore || roscore
    rosparam set /use_sim_time true
    roslaunch neato_node set_urdf.launch
    roslaunch my_localizer test_builtin.launch test_case:=ac109_1
    #roslaunch neato_2dnav amcl_builtin.launch map_file:=/home/jspear/catkin_ws/src/robot_localization_2017/my_localizer/maps/ac109_1.yaml 
    #rosbag play --clock ../bags/ac109_1.bag
    echo "hi" # execute a command as root for which the password is automatically filled
    #$SHELL # keep the terminal open after the previous commands are executed

}

export -f commands

gnome-terminal -e "bash -c 'commands'"