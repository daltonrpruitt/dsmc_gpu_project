#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
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
  int total_spots = -1;

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

  void set_raw_device_pointers(int inlet_particles) {
    raw_pointers.px = thrust::raw_pointer_cast(d_pos_x.data());
    raw_pointers.py = thrust::raw_pointer_cast(d_pos_y.data());
    raw_pointers.pz = thrust::raw_pointer_cast(d_pos_z.data());
    raw_pointers.vx = thrust::raw_pointer_cast(d_vel_x.data());
    raw_pointers.vy = thrust::raw_pointer_cast(d_vel_y.data());
    raw_pointers.vz = thrust::raw_pointer_cast(d_vel_z.data());
    raw_pointers.type = thrust::raw_pointer_cast(d_type.data());
    raw_pointers.index = thrust::raw_pointer_cast(d_index.data());
    raw_pointers.size = d_pos_x.size();

    int offset_to_inlet_cells = raw_pointers.size - inlet_particles;

    empty_raw_pointers.px = raw_pointers.px + offset_to_inlet_cells;
    empty_raw_pointers.py = raw_pointers.py + offset_to_inlet_cells;
    empty_raw_pointers.pz = raw_pointers.pz + offset_to_inlet_cells;
    empty_raw_pointers.vx = raw_pointers.vx + offset_to_inlet_cells;
    empty_raw_pointers.vy = raw_pointers.vy + offset_to_inlet_cells;
    empty_raw_pointers.vz = raw_pointers.vz + offset_to_inlet_cells;
    empty_raw_pointers.type = raw_pointers.type + offset_to_inlet_cells;
    empty_raw_pointers.index = raw_pointers.index+ offset_to_inlet_cells;
    empty_raw_pointers.size = inlet_particles;
  }

  void init(int total_particles, int inlet_particles) {
    h_pos_x = host_vector<float>(total_particles, 0);
    h_pos_y = host_vector<float>(total_particles, 0);
    h_pos_z = host_vector<float>(total_particles, 0);
    
    h_vel_x = host_vector<float>(total_particles, 0);
    h_vel_y = host_vector<float>(total_particles, 0);
    h_vel_z = host_vector<float>(total_particles, 0);
   
    h_type = host_vector<int>(total_particles, -1);
    h_index = host_vector<int>(total_particles, -1);

    total_spots = total_particles;
    copy_host_to_device();
    set_raw_device_pointers(inlet_particles);
  }

  particle_gpu_h_d(int total_particles, int inlet_particles) {
    init(total_particles, inlet_particles);
  }

  particle_gpu_h_d(int total_cells, int inlet_cells, int mppc) {
    int total_particles = total_cells*mppc*5/4 + inlet_cells*mppc;
    init(total_particles, inlet_cells*mppc);
  }

  
  particle_gpu_h_d(vector<particle> &in_particles) {
    num_valid_particles = in_particles.size(); 
    for(int i=0; i<num_valid_particles; ++i) {
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

  particle_gpu_h_d& operator= (vector<particle> &in_particles) {
    num_valid_particles = in_particles.size(); 
    for(int i=0; i<num_valid_particles; ++i) {
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

  void resize_for_init_boundaries(int num_inlet_particles){
    if(num_inlet_particles + num_valid_particles < total_spots) return; 

    int new_total_particles = total_spots+num_inlet_particles*5;

    copy_device_vector_to_host();

    // host_vector<float> n_h_pos_x, n_h_pos_y, n_h_pos_z;
    // host_vector<float> n_h_vel_x, n_h_vel_y, n_h_vel_z;
    // host_vector<int> n_h_type ;
    // host_vector<int> n_h_index ;

    h_pos_x.resize(new_total_particles);
    h_pos_y.resize(new_total_particles);
    h_pos_z.resize(new_total_particles);
    h_vel_x.resize(new_total_particles);
    h_vel_y.resize(new_total_particles);
    h_vel_z.resize(new_total_particles);
    h_type.resize(new_total_particles);
    h_index.resize(new_total_particles);

    thrust::fill(h_pos_x.begin()+total_spots, h_pos_x.end(), 0);
    thrust::fill(h_pos_y.begin()+total_spots, h_pos_y.end(), 0);
    thrust::fill(h_pos_z.begin()+total_spots, h_pos_z.end(), 0);
    thrust::fill(h_vel_x.begin()+total_spots, h_vel_x.end(), 0);
    thrust::fill(h_vel_y.begin()+total_spots, h_vel_y.end(), 0);
    thrust::fill(h_vel_z.begin()+total_spots, h_vel_z.end(), 0);
    thrust::fill( h_type.begin()+total_spots,  h_type.end(),-1);
    thrust::fill(h_index.begin()+total_spots, h_index.end(), -1);

    // n_h_pos_x = host_vector<float>(new_total_particles, 0);
    // n_h_pos_y = host_vector<float>(new_total_particles, 0);
    // n_h_pos_z = host_vector<float>(new_total_particles, 0);
    
    // n_h_vel_x = host_vector<float>(new_total_particles, 0);
    // n_h_vel_y = host_vector<float>(new_total_particles, 0);
    // n_h_vel_z = host_vector<float>(new_total_particles, 0);
   
    // n_h_type = host_vector<int>(new_total_particles, -1);
    // n_h_index = host_vector<int>(new_total_particles, 0);

    // thrust::copy(h_pos_x.begin(), h_pos_x.end(), n_h_pos_x.begin());
    // thrust::copy(h_pos_y.begin(), h_pos_y.end(), n_h_pos_y.begin());
    // thrust::copy(h_pos_z.begin(), h_pos_z.end(), n_h_pos_z.begin());
    // thrust::copy(h_vel_x.begin(), h_vel_x.end(), n_h_vel_x.begin());
    // thrust::copy(h_vel_y.begin(), h_vel_y.end(), n_h_vel_y.begin());
    // thrust::copy(h_vel_z.begin(), h_vel_z.end(), n_h_vel_z.begin());
    // thrust::copy( h_type.begin(),  h_type.end(),  n_h_type.begin());
    // thrust::copy(h_index.begin(), h_index.end(), n_h_index.begin());

    // h_pos_x = n_h_pos_x;
    // h_pos_y = n_h_pos_y;
    // h_pos_z = n_h_pos_z;
    // h_vel_x = n_h_vel_x;
    // h_vel_y = n_h_vel_y;
    // h_vel_z = n_h_vel_z;
    // h_type  = n_h_type ;
    // h_index = n_h_index;

    total_spots = new_total_particles;
    copy_host_to_device();
    set_raw_device_pointers(num_inlet_particles);
  }

  vector<particle> slice(int start, int size) {
    copy_device_vector_to_host();
    vector<particle> particles;
    for(int i=start; i < start + size; ++i) {
      if(i >= num_valid_particles) { break; }
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

  vector<particle> to_vector() {
    return slice(0, num_valid_particles);
  }

  void sort_particles_by_validity(){
    thrust::zip_iterator<particlesTuple> particles_iterator_tuple(
      thrust::make_tuple(
        d_pos_x.begin(), d_pos_y.begin(), d_pos_z.begin(), 
        d_vel_x.begin(), d_vel_y.begin(), d_vel_z.begin(),
        d_index.begin()
    ));
    thrust::sort_by_key(d_type.begin(), d_type.end(), particles_iterator_tuple, thrust::greater<int>());
    calc_num_valid_particles();
  }

  void sort_valid_particles_by_index() {
    sort_particles_by_validity();
    // Should have sort_particles_by_validity() here, or assume is part of calling computation...?
    thrust::zip_iterator<particlesTuple> particles_iterator_tuple(
      thrust::make_tuple(
        d_pos_x.begin(), d_pos_y.begin(), d_pos_z.begin(), 
        d_vel_x.begin(), d_vel_y.begin(), d_vel_z.begin(),
        d_type.begin()
    ));
    calc_num_valid_particles();
    thrust::sort_by_key(d_index.begin(), d_index.begin()+num_valid_particles, particles_iterator_tuple);
  }

  void calc_num_valid_particles() {
    int num_invalid_particles_ = thrust::count(d_type.begin(),d_type.end(), -1);
    num_valid_particles = d_type.size() - num_invalid_particles_;
    raw_pointers.num_valid_particles = num_valid_particles;
  }

  vector<particle> get_valid_particles() {
    sort_particles_by_validity();
    return slice(num_valid_particles);
  }

  void print_size() {
    calc_num_valid_particles();
    printf("Size of particles: ");
    printf(" particles=%d ; spots total=%zu \n", num_valid_particles, h_type.size());
  }

  /**
   * Prints a small sample of particles and various data
   *  
   *  version bit set |  data output added
   *         0        |      velocity
   *         1        |      index
   *         2        |      type
   */
  void print_sample(int version=0, int start_index = 0, int offset = 200) {
    // sort_particles_by_validity();
    copy_device_vector_to_host();
    if( start_index < 0) start_index = 0;
    for(int i=0; i<offset*4; i+=offset) {
      if(start_index + i >= total_spots) {printf("   (Array ends at %d )\n", total_spots); break;}
      for(int j=0; j<6; ++j) {
        int idx = start_index + i + j;
        if(idx >= total_spots){break;}
        printf("%4d:", idx);
        if(h_type[idx] != -1)
          printf("(%1.5f,%1.5f,%1.5f)", h_pos_x[idx], h_pos_y[idx], h_pos_z[idx]);
        else
          printf("     Invalid Particle     ");
        printf(" |");
      }
      if(0b0001 & version){ 
        printf("\n  Vel:");
        for(int j=0; j<6; ++j) {
          int idx = start_index + i + j;
          if(idx >= total_spots){break;}
          printf("   ");
          if(h_type[idx] != -1)
            printf("(%1.5f,%1.5f,%1.5f)", h_vel_x[idx], h_vel_y[idx], h_vel_z[idx]);
          else
            printf("           -            ");
          printf("    ");
        }
      }
      if(0b0010 & version){ 
        printf("\n  Index:");
        for(int j=0; j<6; ++j) {
          int idx = start_index + i + j;
          if(idx >= total_spots){break;}
          printf("   ");
          if(h_type[idx] != -1)
            printf("         %06d           ", h_index[idx]);
          else
            printf("           -            ");
          printf("    ");
        }
      }
      if(0b0100 & version){ 
        printf("\n   Type:");
        for(int j=0; j<6; ++j) {
          int idx = start_index + i + j;
          if(idx >= total_spots){break;}
          printf("   ");
          if(h_type[idx] != -1)
            printf("         %06d           ", h_type[idx]);
          else
            printf("           -            ");
          printf("    ");
        }
      }
      printf("\n");
    }
    printf("\n");
  }

  void dump(bool only_valid = true) {

    printf("Dumping all ");
    if(only_valid) printf("valid ");
    printf("particles:\n");
    int spots = only_valid ? num_valid_particles : total_spots;
    for(int i=0; i<spots; i+=6*4) {
      print_sample(1+2+4,i,6);
    }
  }
} ;

struct particle_count_map_gpu_raw {
  int *cell_idxs = nullptr, *particle_counts = nullptr, 
      *particle_offsets = nullptr;
  int num_occupied_cells = -1;
};

struct particle_count_map {
  device_vector<int> cell_idxs;
  device_vector<int> particle_counts;
  device_vector<int> particle_offsets;

  particle_count_map_gpu_raw raw_pointers;
  int num_cells = -1;
  int num_occupied_cells = -1;

  particle_count_map(int num_cells_) {
    num_cells = num_cells_;
    cell_idxs = device_vector<float>(num_cells, 0);
    particle_counts = device_vector<float>(num_cells, -1);
    particle_offsets = device_vector<float>(num_cells, 0);
  }


  int map_particles_to_cells(particle_gpu_h_d& particles) {
    particles.sort_valid_particles_by_index(); // this is necessary since reduce_by_key() only reduces contiguously

    thrust::pair<IntIter, IntIter> map_ends;

    map_ends = thrust::reduce_by_key(thrust::device,
      particles.d_index.begin(), particles.d_index.end(), thrust::make_constant_iterator(1),
      cell_idxs.begin(), particle_counts.begin());
    
    thrust::exclusive_scan(particle_counts.begin(), map_ends.second, particle_offsets.begin()); 
    num_occupied_cells = map_ends.first - cell_idxs.begin()-1;
    if(num_occupied_cells > num_cells) {
      printf("Error: occupied_cells is larger than actual cells in simulation! (%d > %d)\n", num_occupied_cells, num_cells);
      printf("Dumping values:\n");
      particles.dump();
      print_sample(true);
      return -1;
    }
    set_raw_pointers();
    return 0;
  }
   
  void set_raw_pointers() {
    raw_pointers.num_occupied_cells = num_occupied_cells; 
    raw_pointers.cell_idxs = thrust::raw_pointer_cast(cell_idxs.data());
    raw_pointers.particle_counts = thrust::raw_pointer_cast(particle_counts.data());
    raw_pointers.particle_offsets = thrust::raw_pointer_cast(particle_offsets.data());
  }

  void print_size() {
    printf("Num of cells with particles = %d\n", num_occupied_cells);
  }

  void print_sample(bool dump=false) {
    host_vector<int> h_idxs(cell_idxs.size()), h_counts(particle_counts.size()), 
                     h_offsets(particle_offsets.size());
    thrust::copy(cell_idxs.begin(), cell_idxs.end(), h_idxs.begin());
    thrust::copy(particle_counts.begin(), particle_counts.end(), h_counts.begin());
    thrust::copy(particle_offsets.begin(), particle_offsets.end(), h_offsets.begin());

    int limit = dump ? num_cells : 40; 
    printf("Particle Counts: (Idx:Count)\n");
    for(int i=0; i<limit; i+=10) {
      if(i >= num_cells) {printf("   (End of valid values)\n"); break;}
      printf("   ");
      for(int j=0; j<10; ++j) {
        int idx = i + j;
        if(idx >= num_cells) {break;}
        if(idx < num_occupied_cells)
          printf("%5d:%-5d|",h_idxs[idx],h_counts[idx]);
        else  
          printf("  -  :  -  |");
        
      }
      printf("\n");
    }
    printf("Particle Offsets: (Idx:Offset)\n");
    for(int i=0; i<limit; i+=10) {
      if(i >= num_cells) {printf("   (End of valid values)\n"); break;}
      printf("   ");
      for(int j=0; j<10; ++j) {
        int idx = i + j;
        if(idx >= num_cells) {break;}
        if(idx < num_occupied_cells)
          printf("%5d:%-5d|",h_idxs[idx],h_offsets[idx]);
        else  
          printf("  -  :  -  |");
      }
      printf("\n");
    }

  }
};



struct cellSample_gpu_raw {
  unsigned int *nparticles = nullptr;
  float *vx = nullptr, *vy = nullptr, *vz = nullptr,
        *energy = nullptr;
  int num_cells = -1;

};

// Information that is sampled from the particles to the cells over
// several timesteps
struct cellSample_gpu {
  host_vector<unsigned int> h_nparticles ;            // total number of particles sampled
  host_vector<float> h_vel_x, h_vel_y, h_vel_z;  // total velocity vector
  host_vector<float> h_energy ;                // total kinetic energy of particles

  device_vector<unsigned int> d_nparticles ;            // total number of particles sampled
  device_vector<float> d_vel_x, d_vel_y, d_vel_z;  // total velocity vector
  device_vector<float> d_energy ;                // total kinetic energy of particles

  cellSample_gpu_raw raw_pointers; 

  int num_cells;
  
  void copy_host_to_device() {
    d_nparticles = h_nparticles;

    d_vel_x = h_vel_x;
    d_vel_y = h_vel_y;
    d_vel_z = h_vel_z;

    d_energy = h_energy;
  }

  void set_raw_device_pointers() {
    raw_pointers.nparticles = thrust::raw_pointer_cast(d_nparticles.data());
    raw_pointers.vx = thrust::raw_pointer_cast(d_vel_x.data());
    raw_pointers.vy = thrust::raw_pointer_cast(d_vel_y.data());
    raw_pointers.vz = thrust::raw_pointer_cast(d_vel_z.data());
    raw_pointers.energy = thrust::raw_pointer_cast(d_energy.data());
    raw_pointers.num_cells = num_cells;
  }

  cellSample_gpu(int total_cells) {
    num_cells = total_cells;
    h_nparticles = host_vector<unsigned int>(total_cells, 0);
    h_vel_x = host_vector<float>(total_cells, 0);
    h_vel_y = host_vector<float>(total_cells, 0);
    h_vel_z = host_vector<float>(total_cells, 0);
    h_energy = host_vector<float>(total_cells, 0);
    copy_host_to_device();
    set_raw_device_pointers();

  }

  cellSample_gpu operator= (vector<cellSample> &in_samples) {
    unsigned long size = in_samples.size(); 
    for(long unsigned int i=0; i<size; ++i) {
      h_nparticles[i] = in_samples[i].nparticles;
      h_vel_x[i] = in_samples[i].vel.x;
      h_vel_y[i] = in_samples[i].vel.y;
      h_vel_z[i] = in_samples[i].vel.z;
      h_energy[i] = in_samples[i].energy;
    }
    copy_host_to_device();
    return *this;
  }

  void copy_device_to_host() {
    thrust::copy(d_nparticles.begin(), d_nparticles.end(), h_nparticles.begin());
    thrust::copy(d_vel_x.begin(), d_vel_x.end(), h_vel_x.begin());
    thrust::copy(d_vel_y.begin(), d_vel_y.end(), h_vel_y.begin());
    thrust::copy(d_vel_z.begin(), d_vel_z.end(), h_vel_z.begin());
    thrust::copy(d_energy.begin(), d_energy.end(), h_energy.begin());
  }

  vector<cellSample> slice(int start, int size) {
    copy_device_to_host();
    vector<cellSample> samples;
    for(int i=start; i < start + size; ++i) {
      cellSample s;
      s.nparticles = h_nparticles[i];
      s.vel = vect3d(h_vel_x[i],h_vel_y[i],h_vel_z[i]);
      s.energy = h_energy[i];

      samples.push_back(s);
    }
    return samples;
  }

  vector<cellSample> slice(int size) {
    return slice(0, size);
  }

  vector<cellSample> to_vector() {
    return slice(0, num_cells);
  }

  void print_sample(int start_index = 0, int offset = 5) {
    if(start_index >= num_cells) {
      printf("Error in cellSample_gpu.print_sample() call!\n");
      return;
    }
    copy_device_to_host();
    if( start_index < 0) start_index = 0;
    printf("idx: #particles,vel(x,y,z),energy\n");
    for(int i=start_index; i<start_index+offset*4; i+=4) {
      for(int j=0; j<4; ++j) {
        int idx = i + j;
        if(idx >= num_cells) break;
        printf("%4d: %2d,(%1.1e,%1.1e,%1.1e),%1.1e |",
          idx,h_nparticles[idx],h_vel_x[idx],h_vel_y[idx],h_vel_z[idx],h_energy[idx]);
      }
      printf("\n");
      if(i >= num_cells) {printf("   (End of valid values)\n"); return;}
    }
    printf("\n");
  }

  void dump() {
    printf("Dumping all Cell Samples:\n");
    for(int i=0; i<num_cells; i+=10*4) {
      print_sample(i,10);
    }
  }


} ;

struct collisionInfo_gpu_raw {
  float *maxCollisionRate = nullptr, *collisionRemainder = nullptr;
  int num_cells = -1;
};

// Information that is used to control the collision probability code
struct collisionInfo_gpu {
  // Maximum collision rate seen for this cell so far in the simulation
  host_vector<float> h_maxCollisionRate ;
  // Non-integral fraction of collisions that remain to be performed
  // and are carried over into the next timestep
  host_vector<float> h_collisionRemainder ;

  device_vector<float> d_maxCollisionRate ;
  device_vector<float> d_collisionRemainder ;

  collisionInfo_gpu_raw raw_pointers;
  int num_cells;

  void copy_host_to_device() {
    d_maxCollisionRate = h_maxCollisionRate;
    d_collisionRemainder = h_maxCollisionRate;
  }

  void set_raw_device_pointers() {
    raw_pointers.maxCollisionRate = thrust::raw_pointer_cast(d_maxCollisionRate.data());
    raw_pointers.collisionRemainder = thrust::raw_pointer_cast(d_collisionRemainder.data());
    raw_pointers.num_cells = num_cells;
  }

  collisionInfo_gpu(int total_cells) {
    h_maxCollisionRate = host_vector<float>(total_cells, 0);
    h_collisionRemainder = host_vector<float>(total_cells, 0);
    num_cells = total_cells;
    copy_host_to_device();
    set_raw_device_pointers();
  }

  collisionInfo_gpu operator= (vector<collisionInfo> &in_info) {
    unsigned long size = in_info.size(); 
    for(long unsigned int i=0; i<size; ++i) {
      h_maxCollisionRate[i] = in_info[i].maxCollisionRate;
      h_collisionRemainder[i] = in_info[i].collisionRemainder;
    }
    copy_host_to_device();
    return *this;
  }

  void copy_device_to_host() {
    thrust::copy(d_maxCollisionRate.begin(), d_maxCollisionRate.end(), h_maxCollisionRate.begin());
    thrust::copy(d_collisionRemainder.begin(), d_collisionRemainder.end(), h_collisionRemainder.begin());
  }

  void print_sample(int start_index = 0, int offset = 5) {
    if(start_index >= num_cells) {
      printf("Error in collissionInfo_gpu.print_sample() call!\n");
      return;
    }

    copy_device_to_host();
    if( start_index < 0) start_index = 0;
    printf("idx: maxCollisionRate(*10^27),collisionRemainder\n");
    for(int i=start_index; i<start_index+offset*4; i+=4) {
      for(int j=0; j<4; ++j) {
        int idx = i + j;
        if(idx >= num_cells){break;}
        printf("%4d: %1.5f,%1.5f |",idx,h_maxCollisionRate[idx]*1e27,h_collisionRemainder[idx]);
      }
      printf("\n");
      if(i + 4 >= num_cells) {printf("   (End of valid values)\n"); return;}
    }
    printf("\n");
  }

  void dump() {
    printf("Dumping all Collisions Info:\n");
    for(int i=0; i<num_cells; i+=10*4) {
      print_sample(i,10);
    }
  }

} ;
