#include "structs.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

using thrust::device_vector;


// Data structure for holding particle information
struct particle_gpu {
  // vect3d pos,vel ; // particle position and velocity
  device_vector<float> pos_x, pos_y, pos_z;
  device_vector<float> vel_x, vel_y, vel_z;
  
  device_vector<int> type ;       // Collision status
                                  // 0 inflow particle
                                  // 1 collision with particle that collided with body
                                  // 2 collided with body
  device_vector<int> index ;      // index of containing cell

} ;

// Information that is sampled from the particles to the cells over
// several timesteps
struct cellSample_gpu {
  device_vector<int> nparticles ;            // total number of particles sampled
  device_vector<float> vel_x, vel_y, vel_z;  // total velocity vector
  device_vector<int> energy ;                // total kinetic energy of particles
} ;

// Information that is used to control the collision probability code
struct collisionInfo_gpu {
  // Maximum collision rate seen for this cell so far in the simulation
  device_vector<float> maxCollisionRate ;
  // Non-integral fraction of collisions that remain to be performed
  // and are carried over into the next timestep
  device_vector<float> collisionRemainder ;
} ;
