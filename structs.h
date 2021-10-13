#ifndef STRUCTS_H
#define STRUCTS_H


#include "vect3d.h"

// Data structure for holding particle information
struct particle {
  vect3d pos,vel ; // particle position and velocity
  int type ;       // Collision status
                   // 0 inflow particle
                   // 1 collision with particle that collided with body
                   // 2 collided with body
  int index ;      // index of containing cell
} ;

// Information that is sampled from the particles to the cells over
// several timesteps
struct cellSample {
  int nparticles ; // total number of particles sampled
  vect3d vel ;     // total velocity vector
  float energy ;   // total kinetic energy of particles
} ;

// Information that is used to control the collision probability code
struct collisionInfo {
  // Maximum collision rate seen for this cell so far in the simulation
  float maxCollisionRate ;
  // Non-integral fraction of collisions that remain to be performed
  // and are carried over into the next timestep
  float collisionRemainder ;
} ;

#endif