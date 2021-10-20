#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include "structs.h"
#include "vect3d.h"

using std::vector;
using thrust::host_vector;
using thrust::device_vector;

typedef thrust::device_vector<float>::iterator FloatIter;
typedef thrust::device_vector<int>::iterator   IntIter;
typedef thrust::tuple<FloatIter, FloatIter, FloatIter, 
                      FloatIter, FloatIter, FloatIter, 
                      IntIter> 
    particlesTuple;


struct particle_gpu_raw {
  float *px = nullptr, *py = nullptr, *pz = nullptr, 
        *vx = nullptr, *vy = nullptr, *vz = nullptr;
  int *type = nullptr, *index = nullptr;
  long unsigned int size = 0;
  long unsigned int num_valid_particles = 0;
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
  particle_gpu_raw empty_raw_pointers;

  int num_valid_particles = -1;

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

  void set_raw_device_pointers(int inlet_cells, int mppc) {
    raw_pointers.px = thrust::raw_pointer_cast(d_pos_x.data());
    raw_pointers.py = thrust::raw_pointer_cast(d_pos_y.data());
    raw_pointers.pz = thrust::raw_pointer_cast(d_pos_z.data());
    raw_pointers.vx = thrust::raw_pointer_cast(d_vel_x.data());
    raw_pointers.vy = thrust::raw_pointer_cast(d_vel_y.data());
    raw_pointers.vz = thrust::raw_pointer_cast(d_vel_z.data());
    raw_pointers.type = thrust::raw_pointer_cast(d_type.data());
    raw_pointers.index = thrust::raw_pointer_cast(d_index.data());
    raw_pointers.size = d_pos_x.size();

    int offset_to_inlet_cells = raw_pointers.size - inlet_cells*mppc;

    empty_raw_pointers.px = raw_pointers.px + offset_to_inlet_cells;
    empty_raw_pointers.py = raw_pointers.py + offset_to_inlet_cells;
    empty_raw_pointers.pz = raw_pointers.pz + offset_to_inlet_cells;
    empty_raw_pointers.vx = raw_pointers.vx + offset_to_inlet_cells;
    empty_raw_pointers.vy = raw_pointers.vy + offset_to_inlet_cells;
    empty_raw_pointers.vz = raw_pointers.vz + offset_to_inlet_cells;
    empty_raw_pointers.type = raw_pointers.type + offset_to_inlet_cells;
    empty_raw_pointers.index = raw_pointers.index+ offset_to_inlet_cells;
    empty_raw_pointers.size = inlet_cells*mppc;
  }

  particle_gpu_h_d(int total_cells, int inlet_cells, int mppc) {
    int total_particles = (total_cells + inlet_cells)*mppc;
    h_pos_x = host_vector<float>(total_particles, 0);
    h_pos_y = host_vector<float>(total_particles, 0);
    h_pos_z = host_vector<float>(total_particles, 0);
    
    h_vel_x = host_vector<float>(total_particles, 0);
    h_vel_y = host_vector<float>(total_particles, 0);
    h_vel_z = host_vector<float>(total_particles, 0);
   
    h_type = host_vector<int>(total_particles, -1);
    h_index = host_vector<int>(total_particles, 0);
    
    copy_host_to_device();
    set_raw_device_pointers(inlet_cells, mppc);
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

  void copy_device_vector_to_host() {
    thrust::copy(d_pos_x.begin(), d_pos_x.end(), h_pos_x.begin());
    thrust::copy(d_pos_y.begin(), d_pos_y.end(), h_pos_y.begin());
    thrust::copy(d_pos_z.begin(), d_pos_z.end(), h_pos_z.begin());
    thrust::copy(d_vel_x.begin(), d_vel_x.end(), h_vel_x.begin());
    thrust::copy(d_vel_y.begin(), d_vel_y.end(), h_vel_y.begin());
    thrust::copy(d_vel_z.begin(), d_vel_z.end(), h_vel_z.begin());
    thrust::copy(d_type.begin() , d_type.end() ,  h_type.begin());
    thrust::copy(d_index.begin(), d_index.end(), h_index.begin());
  }

  vector<particle> slice(int start, int size) {
    copy_device_vector_to_host();
    vector<particle> particles;
    for(int i=start; i < start + size; ++i) {
      particle p;
      p.pos = vect3d(h_pos_x[i],h_pos_y[i],h_pos_z[i]);
      p.vel = vect3d(h_vel_x[i],h_vel_y[i],h_vel_z[i]);
      p.type = h_type[i];
      p.index = h_index[i];

      particles.push_back(p);
    }
    return particles;
  }

  vector<particle> slice(int size) {
    return slice(0, size);
  }

  void sort_particles_by_validity(){
    thrust::zip_iterator<particlesTuple> particles_iterator_tuple(
      thrust::make_tuple(
        d_pos_x.begin(), d_pos_y.begin(), d_pos_z.begin(), 
        d_vel_x.begin(), d_vel_y.begin(), d_vel_z.begin(),
        d_index.begin()
    ));
    thrust::sort_by_key(d_type.begin(), d_type.end(), particles_iterator_tuple, thrust::greater<int>());
    int num_invalid_particles_ = thrust::count(d_type.begin(),d_type.end(), -1);
    num_valid_particles = d_type.size() - num_invalid_particles_;
    raw_pointers.num_valid_particles = num_valid_particles;
  }

  vector<particle> get_valid_particles() {
    sort_particles_by_validity();
    return slice(num_valid_particles);
  }

  void print_size() {
    sort_particles_by_validity();
    printf("Size of particles: ");
    printf(" particles=%d ; spots total=%zu \n", num_valid_particles, h_type.size());
  }

  void print_small_sample(int version=0) {
    sort_particles_by_validity();
    copy_device_vector_to_host();
    for(int i=0; i<1000; i+=200 ) {
      for(int j=0; j<6; ++j) {
        int idx = i + j;
        printf("%4d:", idx);
        if(h_type[idx] != -1)
          printf("(%1.5f,%1.5f,%1.5f)", h_pos_x[idx], h_pos_y[idx], h_pos_z[idx]);
        else
          printf("     Invalid Particle     ");
        printf(" |");
      }
      if(version > 0){ 
        printf("\n  Vel:");
        for(int j=0; j<6; ++j) {
          int idx = i + j;
          printf("   ");
          if(h_type[idx] != -1)
            printf("(%1.5f,%1.5f,%1.5f)", h_vel_x[idx], h_vel_y[idx], h_vel_z[idx]);
          else
            printf("    Invalid Particle    ");
          printf("    ");
        }
      }
      printf("\n");
    }
    printf("\n");
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
