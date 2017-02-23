# Robot Localization

The goal of this project was to develop a Particle Filter to localize a neato vacuum cleaner robot. Particle filters can be used in any situation which you have a map of the area that your robot is moving in, but you're not exactly sure where on that map your robot is (as is true when using odometry). By dropping a bunch of points randomly on the map, we can calculate how closeley a hypothetical senor on that point lines up with the sensor data we are recieving from our robot. Using this method, we can accurately pinpoint where our robot is on the map. For a more in-depth explanation see [here](https://en.wikipedia.org/wiki/Particle_filter).

The majority of the code lives in pf.py. It is broken down into four major steps:

1. Initializing the particles randomly on the map.
2. Updating the positions of each particle using data from odometry
3. Use the laser scan data to reweight each particle.
4. Resample the particles with probability proportional to each particle's weight.

You can see the particle filter running for two different maps below:

![Localizer In Action in AC109](https://github.com/Joboman555/robot_localization_2017/blob/master/my_localizer/resources/109_1.gif)


![Localizer In Action in AC109_2](https://github.com/Joboman555/robot_localization_2017/blob/master/my_localizer/resources/109_2.gif)

The big red arrow is the posiion of the robot according to odometry. The black circle is the particle filter's guess at where the robot is. Each particle is represented by a little arrow, with size and color proportional to its weight.

### Design

One design decision I am proud of is the way in which I illustrated the particles. Originally, the particles were all illustrated as 2d red arrows. By switching from arrows to vectors, I was able to change the color and length of the particles based on their weight. Being able to get a visual cue of the weight of each particle made debugging much easier while building the codebase.

### Challenges

A major challenge in this project was dealing with the growing complexity. Over the course of the project, the codebase increased from ~200 to ~600 LOC (as of 2/23/17), so atomizing functionality with functions was paramount to keeping the code readable and making progress.

### Going Further

If I had more time, there are a lot of places I would like to take this project. First of all, much can be done in terms of vectorization and parallelizing the computations done to determine the weight of the particles. Increasing the speed would make it possible to bump up the particle count, increasing the accuracy of the particle filter. The next major step would be creating a wrapper function that ran the particle filter with varying input parameters, using either a search or a gradient descent to automatically tune the filter.
