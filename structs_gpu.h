#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "structs.h"
#include "vect3d.h"

using std::vector;
using thrust::host_vector;
using thrust::device_vector;

struct particle_gpu_raw {
  float *px, *py, *pz, *vx, *vy, *vz;
  int *type, *index;
};

// Data structure for holding particle information
struct particle_gpu_h_d {
  // vect3d pos,vel ; // particle position and velocity

  host_vector<float> h_pos_x, h_pos_y, h_pos_z;
  host_vector<float> h_vel_x, h_vel_y, h_vel_z;
  
  host_vector<int> h_type ;       // Collision status
                                  // -1 Uninitialized particle
                                  // 0 inflow particle
                                  // 1 collision with particle that collided with body
                                  // 2 collided with body
  host_vector<int> h_index ;      // index of containing cell
  
  device_vector<float> d_pos_x, d_pos_y, d_pos_z;
  device_vector<float> d_vel_x, d_vel_y, d_vel_z;
  
  device_vector<int> d_type ;       
  device_vector<int> d_index ; 

  particle_gpu_raw raw_pointers;

  void copy_host_to_device() {
    d_pos_x = h_pos_x;
    d_pos_y = h_pos_y;
    d_pos_z = h_pos_z;
 
    d_vel_x = h_vel_x;
    d_vel_y = h_vel_y;
    d_vel_z = h_vel_z;

    d_type = h_type;
    d_index = h_index;

  }

  void set_raw_device_pointers() {
    raw_pointers.px = thrust::raw_pointer_cast(d_pos_x.data());
    raw_pointers.py = thrust::raw_pointer_cast(d_pos_y.data());
    raw_pointers.pz = thrust::raw_pointer_cast(d_pos_z.data());
    raw_pointers.vx = thrust::raw_pointer_cast(d_vel_x.data());
    raw_pointers.vy = thrust::raw_pointer_cast(d_vel_y.data());
    raw_pointers.vz = thrust::raw_pointer_cast(d_vel_z.data());
    raw_pointers.type = thrust::raw_pointer_cast(d_type.data());
    raw_pointers.index = thrust::raw_pointer_cast(d_index.data());
  }

  particle_gpu_h_d(long unsigned int size) {
    h_pos_x = host_vector<float>(size, 0);
    h_pos_y = host_vector<float>(size, 0);
    h_pos_z = host_vector<float>(size, 0);
    
    h_vel_x = host_vector<float>(size, 0);
    h_vel_y = host_vector<float>(size, 0);
    h_vel_z = host_vector<float>(size, 0);
   
    h_type = host_vector<int>(size, -1);
    h_index = host_vector<int>(size, 0);
    
    copy_host_to_device();
    set_raw_device_pointers();
  }
  particle_gpu_h_d(vector<particle> &in_particles) {
    unsigned long size = in_particles.size(); 
    for(long unsigned int i=0; i<size; ++i) {
      h_pos_x[i] = in_particles[i].pos.x;
      h_pos_y[i] = in_particles[i].pos.y;
      h_pos_z[i] = in_particles[i].pos.z;

      h_vel_x[i] = in_particles[i].vel.x;
      h_vel_y[i] = in_particles[i].vel.y;
      h_vel_z[i] = in_particles[i].vel.z;

      h_type[i] = in_particles[i].type;
      h_index[i] = in_particles[i].index;
    }
    copy_host_to_device();
  }

  particle_gpu_h_d operator= (vector<particle> &in_particles) {
    unsigned long size = in_particles.size(); 
    for(long unsigned int i=0; i<size; ++i) {
      h_pos_x[i] = in_particles[i].pos.x;
      h_pos_y[i] = in_particles[i].pos.y;
      h_pos_z[i] = in_particles[i].pos.z;

      h_vel_x[i] = in_particles[i].vel.x;
      h_vel_y[i] = in_particles[i].vel.y;
      h_vel_z[i] = in_particles[i].vel.z;

      h_type[i] = in_particles[i].type;
      h_index[i] = in_particles[i].index;
    }
    copy_host_to_device();
    return *this;
  }

  vector<particle> device_vector_to_stl_vector() {
    thrust::copy(d_pos_x.begin(), d_pos_x.end(), h_pos_x.begin());
    thrust::copy(d_pos_y.begin(), d_pos_y.end(), h_pos_y.begin());
    thrust::copy(d_pos_z.begin(), d_pos_z.end(), h_pos_z.begin());
    thrust::copy(d_vel_x.begin(), d_vel_x.end(), h_vel_x.begin());
    thrust::copy(d_vel_y.begin(), d_vel_y.end(), h_vel_y.begin());
    thrust::copy(d_vel_z.begin(), d_vel_z.end(), h_vel_z.begin());
    thrust::copy(d_type.begin(),  d_type.end(),  h_type.begin() );
    thrust::copy(d_index.begin(), d_index.end(), h_index.begin());
    vector<particle> particles;
    for(long unsigned int i=0; i < h_pos_x.size(); ++i) {
      particle p;
      p.pos = vect3d(h_pos_x[i],h_pos_y[i],h_pos_z[i]);
      p.vel = vect3d(h_vel_x[i],h_vel_y[i],h_vel_z[i]);
      p.type = h_type[i];
      p.index = h_index[i];

      particles.push_back(p);
    }
    return particles;
  }

  void print_size() {
    printf("Size of particles at %p :\n", (void *)this);
    printf(" particles=%zu ; bytes total=%zu \n", h_pos_x.size(), 
      h_pos_x.size()*sizeof(float)*6 + h_type.size()*sizeof(int)*2 );
  }
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
