#!/usr/bin/env python

from pf import normalize_particles, Particle

class NormalizeParticlesTest:
    def __init__(self):
        pass

    def test(self, input, output):
        return output == normalize_particles(input)

    def empty_list(self):
        particles = []
        res = self.test(particles, [])
        if res:
            return 'Empty List Test Passed!'
        else:
            return 'Empty List Test Failed!'

    def all_zeroes(self):
        particles = [Particle(w=0), Particle(w=0), Particle(w=0)]
        res = self.test(particles, [Particle(w=0), Particle(w=0), Particle(w=0)])
        if res:
            return 'All Zeroes Test Passed!'
        else:
            return 'All Zeros Test Failed!'

    def basic_reweight(self):
        particles = [Particle(w=1.), Particle(w=2.), Particle(w=3.)]
        expected_output = [Particle(w=1./6.), Particle(w=2./6.), Particle(w=1./2.)]
        res = self.test(particles, expected_output)
        if res:
            return 'All Zeroes Test Passed!'
        else:
            return 'All Zeros Test Failed!'

if __name__ == '__main__':
    test_normalize_particles = NormalizeParticlesTest()
    print test_normalize_particles.empty_list()
    print test_normalize_particles.all_zeroes()
    print test_normalize_particles.basic_reweight()