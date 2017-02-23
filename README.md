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

