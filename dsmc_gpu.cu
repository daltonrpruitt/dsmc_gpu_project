#include <iostream>
#include <fstream>
#include <cmath>
// #include <list>
#include <vector>
#include <stdlib.h>

#include "vect3d_gpu.h"
#include "structs_gpu.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/count.h>

#include <curand.h>
#include <curand_kernel.h>

#define DEBUG

// Physical constant describing atom collision size
const float sigmak = 1e-28 ; // collision cross section

// Note, pnum recomputed from mean particle per cell and density
float pnum = 1e27 ; // number of particles per simulated particle

// For this example, the geometry is a simple plate
float plate_x = -0.25 ;
float plate_dy = 0.25 ;
float plate_dz = .5 ;
struct Plate {
  float x=plate_x, dy=plate_dy, dz=plate_dz;

};
Plate plate;

using namespace std ;
using thrust::host_vector;
using thrust::device_vector;
using thrust::raw_pointer_cast;


void cudaErrChk(cudaError_t status, const string &msg, bool &pass)
{
  if (status != cudaSuccess)
  {
    printf("Error with %s! : ", msg.c_str());
    printf("%s\n", cudaGetErrorString(status));
    pass = false;
  }
}

// Driver for the random number function
inline double ranf() {
  return drand48() ;
}

// Computes a unit vector with a random orientation and uniform distribution
__device__
inline vect3d randomDir(float rand1, float rand2) {
  double B = 2. * rand1-1;
  double A = sqrt(1.-B*B);
  double theta = rand2*2*M_PI;
  return vect3d(B,A*cos(theta),A*sin(theta)) ;
}

// Computes a velocity magnitude distribution that would correspond to
// thermal equilibrium
inline double randVel(double vmp) {
  double t = max(ranf(),1e-200) ;
  return vmp*sqrt(-log(t)) ;
}

// Based on https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
__global__ 
void init_rands(long input_seed, curandStatePhilox4_32_10_t *rand4, 
  curandState *rand5, curandState *rand6) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned long long seed = input_seed + idx;
  // __device__ void
  //    curand_init (
  //      unsigned long long seed, unsigned long long subsequence,
  //      unsigned long long offset, curandState_t *state)
  curand_init (seed,   0, 0, &rand4[idx]);
  curand_init (seed+1, 0, 0, &rand5[idx]);
  curand_init (seed+2, 0, 0, &rand6[idx]);
} 

// Create particles at inflow boundary
// This works by creating particles at a ghost cell just before the boundary
// any particles that don't make it into the domain are discarded.
__global__
void initializeBoundaries_gpu(
  particle_gpu_raw particles,
  int ni, int nj, int nk, 
  float vmean, float vtemp, int mppc, 
  curandStatePhilox4_32_10_t *rand4, 
  curandState *rand_5,
  curandState *rand_6) 
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx + mppc - 1 >= particles.size) return;

  double dx=2./float(ni),dy=2./float(nj),dz=2./float(nk) ;
  int j = idx / nj;
  int k = idx * nk;
  double cx = -1-dx ;
  double cy = -1+float(j)*dy ;
  double cz = -1+float(k)*dz ;

  curandStatePhilox4_32_10_t local_rand4 = rand4[idx];
  curandState local_rand_5 = rand_5[idx];
  curandState local_rand_6 = rand_6[idx];

  for(int m=0;m<mppc;++m) { // Compute mcp particles
    float4 rand_results4 = curand_uniform4(&local_rand4);
    float rand_result_5 = curand_uniform(&local_rand_5);
    float rand_result_6 = curand_uniform(&local_rand_6);
    particles.px[idx*mppc + m] = cx + rand_results4.x*dx;
    particles.py[idx*mppc + m] = cy + rand_results4.y*dy;
    particles.pz[idx*mppc + m] = cx + rand_results4.z*dx;

    double t = max(rand_results4.w, 1e-200);
    double speed = vtemp * sqrt(-log(t));

    vect3d vel = randomDir(rand_result_5, rand_result_6);

    particles.vx[idx*mppc + m] = vel.x * speed + vmean;
    particles.vy[idx*mppc + m] = vel.y * speed;
    particles.vz[idx*mppc + m] = vel.z * speed;

    particles.type[idx*mppc + m] = 0;
    particles.index[idx*mppc + m] = 0;
  }
  // write local state back to global
  rand4[idx] = local_rand4;
  rand_5[idx] = local_rand_5;
  rand_6[idx] = local_rand_6;
}


// Move particle for the timestep.  Also handle side periodic boundary
// conditions and specular reflections off of a plate 
__global__
void moveParticlesWithBCs_gpu(particle_gpu_raw particles, float deltaT, Plate plate) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= particles.num_valid_particles) return;

  vect3d pos = vect3d(particles.px[idx], particles.py[idx], particles.pz[idx]);
  vect3d vel = vect3d(particles.vx[idx], particles.vy[idx], particles.vz[idx]);
  vect3d npos = pos + vel*deltaT;
  // Check if particle hits the plate
  if((pos.x < plate.x && npos.x > plate.x) ||
      (pos.x > plate.x && npos.x < plate.x)) {
    // It passes through the plane of the plate, now
    // check if it actually hits the plate
    double t = (pos.x-plate.x)/(pos.x-npos.x) ; // fraction of timestep to hit plate
    vect3d pt = pos*(1.-t)+npos*t ; // interpolated position at time t
    if((pt.y < plate.dy && pt.y > -plate.dy) &&
        (pt.z < plate.dz && pt.z > -plate.dz)) {
          // collides with plate
          // adjust position and velocity (specular reflection)
          npos.x = npos.x - 2*(npos.x-plate.x) ; 
          // why 2 and not some function of t? b/c is calculating new npos, not pos,
          //   so has to get back to plate, then keeps moving the same amount in new new direction
          vel.x = -vel.x ; // Velocity just reflects along x direction
          particles.type[idx] = 2 ;
    }
  }
  // Apply periodic bcs, here we just relocate particle to other side of
  // the domain when it crosses a periodic boundary
  // Note, assuming domain is a unit square
  if(npos.y>1)
    npos.y -= 2.0 ;
  if(npos.y<-1)
    npos.y += 2.0 ;
  if(npos.z>1)
    npos.z -= 2.0 ;
  if(npos.z<-1)
    npos.z += 2.0 ;

  // Update particle positions
  particles.px[idx] = npos.x;
  particles.py[idx] = npos.y;
  particles.pz[idx] = npos.z;
}

// After moving particles, any particles outside of the cells need to be
// discarded as they cannot be indexed.  Since this can happen only at the
// x=-1 or x=1 boundaries, we only need to check the x coordinate
__global__
void removeOutsideParticles_gpu(particle_gpu_raw particles) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= particles.num_valid_particles) return;
    if(particles.px[idx] < -1.0 || particles.px[idx] > 1.0) { // Outside domain so remove
      particles.type[idx] = -1 ;
    }
}


// Compute the cell index of each particle, for a regular Cartesian grid
// this can be computed directly from the coordinates
// Precondition: The device vectors referenced by particles only contains 
//            valid particles (type != -1); this is accomplished by sorting
//            by type and setting the number of valid particles in the 
//            num_valid_particles member of the particle_gpu_raw struct.
__global__
void indexParticles_gpu(particle_gpu_raw particles, int ni, int nj, int nk) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;  
  if(idx >= particles.num_valid_particles) return;

  double dx=2./float(ni),dy=2./float(nj),dz=2./float(nk) ;
  
    // For a Cartesian grid, the mapping from cell to particle is trivial
  int i = min(int(floor((particles.px[idx]+1.0)/dx)),ni-1) ;
  int j = min(int(floor((particles.py[idx]+1.0)/dy)),nj-1) ;
  int k = min(int(floor((particles.pz[idx]+1.0)/dz)),nk-1) ;
  particles.index[idx] = i*nj*nk+j*nk+k ;
}

// Initialize the sampled cell variables to zero
__global__
void initializeSample_gpu(cellSample_gpu_raw cellSample) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;  
  if(idx >= cellSample.num_cells) return;
  cellSample.nparticles[idx] = 0 ;
  cellSample.vx[idx] = 0 ;
  cellSample.vy[idx] = 0 ;
  cellSample.vz[idx] = 0 ;
  cellSample.energy[idx] = 0 ;
}

// Sum particle information to cell samples, this will be used to compute
// collision probabilities
__global__
void sampleParticles_gpu(cellSample_gpu_raw cellData, 
                         particle_gpu_raw particles) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;  
  if(idx >= particles.num_valid_particles) return;
  int index = particles.index[idx] ;
  atomicInc(&cellData.nparticles[index],particles.num_valid_particles) ;
  float vx = particles.vx[idx] ;
  float vy = particles.vy[idx] ;
  float vz = particles.vz[idx] ;
  
  atomicAdd(&cellData.vx[index],vx) ;
  atomicAdd(&cellData.vy[index],vy) ;
  atomicAdd(&cellData.vz[index],vz) ;

  atomicAdd(&cellData.energy[index], .5*(vx*vx + vy*vy + vz*vz) ) ;
} 
  

// Compute particle collisions
__global__
void collideParticles_gpu(particle_gpu_raw particles,
                      collisionInfo_gpu_raw collisionData,
                      cellSample_gpu_raw cellData,
                      int nsample, float cellvol, 
                      float sigmak, float deltaT,
                      float pnum,
                      particle_count_map_gpu_raw mapping,
                      curandStatePhilox4_32_10_t *rand4,
                      curandState *rand5, curandState *rand6) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= mapping.num_occupied_cells) return;
  int cell_idx = mapping.cell_idxs[idx];
  curandStatePhilox4_32_10_t local_rand4 = rand4[idx];
  curandState local_rand_5 = rand5[idx];
  curandState local_rand_6 = rand6[idx];

  // Do not need mapping generation, as have already
  // Loop over cells and select particles to perform collisions
  //for(int i=0;i<ncells;++i) {
  // Compute mean and instantaneous particle numbers for the cell
  float n_mean = float(cellData.nparticles[cell_idx])/float(nsample) ;
  float n_instant = mapping.particle_counts[idx]; //np[i] ;
  // Compute a number of particles that need to be selected for
  // collision tests
  float select = n_instant*n_mean*pnum*collisionData.maxCollisionRate[cell_idx]*deltaT/cellvol + 
        collisionData.collisionRemainder[cell_idx] ;
  // We can only check an integer number of collisions in any timestep
  int nselect = int(select) ;
  // The remainder collision fraction is saved for next timestep
  collisionData.collisionRemainder[cell_idx] = select - float(nselect) ;
  if(nselect > 0) { // selected particles for collision
    if(mapping.particle_counts[idx] < 2) { // if not enough particles for collision, wait until
      // we have enough
      collisionData.collisionRemainder[cell_idx] += nselect ;
    } else {
      float4 rand_results4 = curand_uniform4(&local_rand4);
      float rand_result_5 = curand_uniform(&local_rand_5);
      float rand_result_6 = curand_uniform(&local_rand_6);
  
      // Select nselect particles for possible collision
      float cmax = collisionData.maxCollisionRate[cell_idx] ;
      for(int c=0;c<nselect;++c) {
        // select two points in the cell
        int pt1 = min(int(floor(rand_results4.x*n_instant)),int(n_instant)-1) ;
        int pt2 = min(int(floor(rand_results4.y*n_instant)),int(n_instant)-1) ;

        // Make sure they are unique points
        // This extended version should not be necessarily most of the time, 
        //   as far as I can guess.
        while(pt1==pt2) {
          pt2 = min(int(floor(rand_result_5*n_instant)),int(n_instant)-1) ;
          rand_result_5 = curand_uniform(&local_rand_5);
        }
        // Compute the relative velocity of two particles
        int particle1_idx = mapping.particle_offsets[idx]+pt1;
        float vx1 = particles.vx[particle1_idx];
        float vy1 = particles.vy[particle1_idx];
        float vz1 = particles.vz[particle1_idx];
        vect3d v1 = vect3d(vx1, vy1, vz1);

        int particle2_idx = mapping.particle_offsets[idx]+pt2;
        float vx2 = particles.vx[particle2_idx];
        float vy2 = particles.vy[particle2_idx];
        float vz2 = particles.vz[particle2_idx];
        vect3d v2 = vect3d(vx2, vy2, vz2);

        vect3d vr = v1-v2 ;
        float vrm = norm(vr) ;
        // Compute collision  rate for hard sphere model
        float crate = sigmak*vrm ;
        if(crate > cmax)
          cmax = crate ;
        // Check if these particles actually collide
        if(rand_results4.w < crate/collisionData.maxCollisionRate[cell_idx]) {
          // Collision Accepted, adjust particle velocities
          // Compute center of mass velocity, vcm
          vect3d vcm = .5*(v1+v2) ;
          // Compute random perturbation that conserves momentum
          vect3d vp = randomDir(rand_result_5, rand_result_6)*vrm ;

          // Adjust particle velocities to reflect collision
          vect3d new_v1 = vcm + 0.5*vp;
          vect3d new_v2 = vcm - 0.5*vp;

          particles.vx[particle1_idx] = new_v1.x;
          particles.vy[particle1_idx] = new_v1.y;
          particles.vz[particle1_idx] = new_v1.z;

          particles.vx[particle2_idx] = new_v2.x;
          particles.vy[particle2_idx] = new_v2.y;
          particles.vz[particle2_idx] = new_v2.z;


          // Bookkeeping to track particle interactions
          int t1 = particles.type[particle1_idx] ;
          int t2 = particles.type[particle2_idx] ;
          int tc = (t1+t2>0)?1:0 ;
          particles.type[particle1_idx] = max(tc,t1) ;
          particles.type[particle2_idx] = max(tc,t2) ;
        }
      }
      // Update the maximum collision rate to be used in future timesteps
      // for determining number of particles to select.
      collisionData.maxCollisionRate[cell_idx] = cmax ;
    }
  }
  rand4[idx] = local_rand4;
  rand5[idx] = local_rand_5;
  rand6[idx] = local_rand_6;
}


// Initialize the collision data structures to modest initial values
__global__
void initializeCollision_gpu(collisionInfo_gpu_raw collisionData, 
  float sigmak, float vtemp, curandState *rand) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx >= collisionData.num_cells) return;

  curandState local_rand_state = rand[idx];
  float rand_result = curand_uniform(&local_rand_state);
  rand[idx] = local_rand_state;

  collisionData.maxCollisionRate[idx] = sigmak*vtemp ;
  collisionData.collisionRemainder[idx] = rand_result ;
}


/*****************************************************************************
 **
 ** Simple Direct Simulation Monte Carlo rarefied gas simulator
 ** Runs on cube with corners [1,1,1] and [-1,-1,-1], and a center at the
 ** origin.  x=-1 is an inflow and x=1 is an outflow, all other boundaries
 ** are periodic.
 **
 ** Main program arguments:
 **
 ** -2d:
 **      Switch to two dimensional mode (only one cell in z directions)
 **
 ** -ni:
 **      Number of cells in x direction
 **
 ** -nj:
 **      Number of cells in y direction
 **
 ** -nk:
 **      Number of cells in z direction
 **
 ** -mppc:
 **      Mean Particles per Cell in simulation, adjust number of virtual
 **      particles to meet this target for the inflow
 **
 ** -mach:
 **      Mach number or ratio of mean atom velocity versus mean thermal velocity
 **
 ** -density:
 **      Density of the incoming flow
 **
 ** -platex
 **      x-location of plate
 **
 ** -platedy
 **      y height of plate
 **
 ** -platedz
 **      z width of plate
 **
 ** -time
 **      simulation time step size (usually computed from the above parameters
 **
 *****************************************************************************/

int main(int ac, char *av[]) {
  long seed = 1 ;
  srand48(seed) ;
  int ni=32,nj=32,nk=32 ;
  // mean velocity and temperature of flow
  float Mach = 20 ;
  float vmean=1 ;
  // mean particles per cell
  int mppc = 10 ;
  float density = 1e30 ; // Number of molecules per unit cube of space
  float time = -1 ;
  // Parse command line options
  int i=1;
  while(i<ac && av[i][0] == '-') {
    string opt = string(av[i]) ;
    if(opt == "-2d") {
      nk = 1 ;
      plate.dz = 1 ;
    }
    if(opt == "-ni") 
      ni = atoi(av[++i]) ;
    if(opt == "-nj") 
      nj = atoi(av[++i]) ;
    if(opt == "-nk")
      nk = atoi(av[++i]) ;
    if(opt == "-mppc")
      mppc = atoi(av[++i]) ;
    if(opt == "-mach")
      Mach = atof(av[++i]) ;
    if(opt == "-density")
      density = atof(av[++i]) ;
    if(opt == "-platex")
      plate.x = atof(av[++i]) ;
    if(opt == "-platedy")
      plate.dy = atof(av[++i]) ;
    if(opt == "-platedz")
      plate.dz = atof(av[++i]) ;
    if(opt == "-time") 
      time = atof(av[++i]) ;
    ++i ;
  }

  float vtemp= vmean/Mach ;
  float dx=2./float(ni),dy=2./float(nj),dz=2./float(nk) ;
  double cellvol = dx*dy*dz ;

  // Compute number of molecules a particle represents
  pnum = density*cellvol/float(mppc) ;

  // Create simulation data structures 
  vector<particle> particleVec ;
  particle_gpu_h_d particles(ni*nj*nk, nj*nk, mppc);

  vector<cellSample> cellData(ni*nj*nk) ;
  cellSample_gpu cellData_gpu(ni*nj*nk);


  vector<collisionInfo> collisionData(ni*nj*nk) ;
  collisionInfo_gpu collisionData_gpu(ni*nj*nk);

  // Compute reasonable timestep
  float deltax = 2./float(max(max(ni,nj),nk)) ;
  float deltaT = .1*deltax/(vmean+vtemp) ;

  // If time duration not given, simulate for 4 free-stream flow-through times 
  // DWP: Free stream = the stream that is free from obstacles, i.e. ignoring plate?
  if(time < 0)
    time = 8./(vmean+vtemp) ;

  // Compute nearest power of 2 timesteps
  float tsteps = time/deltaT ;
  int ln2steps = int(ceil(log(tsteps)/log(2.0))) ;
  int ntimesteps = 1<<ln2steps ;
  cout << "time = " << time << ' ' << ", nsteps = " << ntimesteps << endl ;
  int nsample = 0 ;

  // re-sample 4 times during simulation
  const int sample_reset = ntimesteps/4 ;
  int thrds_per_block = 256;

  // curandGenerator_t generator;
  // curandStatus_t curandStatus = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32); // Marsenne
  int num_cells = ni*nj*nk;
  device_vector<curandStatePhilox4_32_10_t> rand4State = 
    device_vector<curandStatePhilox4_32_10_t>(num_cells);
  device_vector<curandState> randState_5 = 
    device_vector<curandState>(num_cells);
  device_vector<curandState> randState_6 = 
    device_vector<curandState>(num_cells);
  
  curandStatePhilox4_32_10_t * rand4State_ptr = raw_pointer_cast(rand4State.data());
  curandState * randState_5_ptr = raw_pointer_cast(randState_5.data());
  curandState * randState_6_ptr = raw_pointer_cast(randState_6.data());
  init_rands<<<thrds_per_block, num_cells/thrds_per_block>>> (
    seed, rand4State_ptr, randState_5_ptr, randState_6_ptr);
    
  // Begin simulation.  Initialize collision data
  // initializeCollision(collisionData,vtemp) ;
  initializeCollision_gpu<<<ni*nj*nk/thrds_per_block+1 ,thrds_per_block>>>(
    collisionData_gpu.raw_pointers,sigmak,vtemp,randState_6_ptr) ;
#ifdef DEBUG
  printf("Collision data num_cells=%d\n",collisionData_gpu.num_cells);
  collisionData_gpu.print_sample();
#endif

  bool pass = true;

#ifdef DEBUG
  particles.print_size();
#endif

  // Step forward in time
  for(int n=0;n<ntimesteps;++n) {
    // Add particles at inflow boundaries
    initializeBoundaries_gpu<<<nj*nk/thrds_per_block, thrds_per_block>>>(
                            particles.empty_raw_pointers,ni,nj,nk,vmean,vtemp,mppc, 
                            rand4State_ptr, randState_5_ptr, randState_6_ptr) ;
    cudaDeviceSynchronize();
    cudaErrChk(cudaGetLastError(), "initializeBoundaries_gpu", pass);
    if(!pass) return -1;
#ifdef DEBUG
    printf("After initializeBoundaries_gpu...\n");
    particles.print_sample(1);    
    particles.print_sample(4, particles.num_valid_particles-12, 6);
#else
    particles.sort_particles_by_validity();
#endif

    int blocks = particles.num_valid_particles / thrds_per_block + 1;
    // Move particles
    moveParticlesWithBCs_gpu<<<blocks, thrds_per_block>>>(particles.raw_pointers,deltaT,plate) ;
    cudaDeviceSynchronize();
    cudaErrChk(cudaGetLastError(), "moveParticlesWithBCs_gpu", pass);
    if(!pass) return -1;

#ifdef DEBUG
    printf("After moveParticlesWithBCs...\n");
    particles.print_sample();
#endif

    // particleVec = particles.get_valid_particles();
    // Remove any particles that are now outside of boundaries
    removeOutsideParticles_gpu<<<blocks, thrds_per_block>>>(particles.raw_pointers) ;
    cudaDeviceSynchronize();
    cudaErrChk(cudaGetLastError(), "removeOutsideParticles_gpu", pass);
    if(!pass) return -1;

#ifdef DEBUG
    printf("After removeOutsideParticles...\n");
    particles.print_size();
    particles.print_sample();
#endif

    particles.sort_particles_by_validity();
#ifdef DEBUG
    printf("Num Blocks for indexParticles: %d\n",particles.num_valid_particles/ thrds_per_block+1);
#endif
    // Compute cell index for particles based on their current
    // locations
    indexParticles_gpu<<<particles.num_valid_particles/thrds_per_block+1, thrds_per_block>>>(
      particles.raw_pointers,ni,nj,nk) ;
    cudaDeviceSynchronize();
    cudaErrChk(cudaGetLastError(), "indexParticles_gpu", pass);
    if(!pass) return -1;

#ifdef DEBUG
    printf("After indexParticles...\n");
    particles.print_size();
    particles.print_sample(2);

#endif

    // If time to reset cell samples, reinitialize data
    if(n%sample_reset == 0 ) {
      initializeSample_gpu<<<ni*nj*nk/thrds_per_block+1 ,thrds_per_block>>>(cellData_gpu.raw_pointers) ;
      nsample = 0 ;
      cudaDeviceSynchronize();
      cudaErrChk(cudaGetLastError(), "initializeSample_gpu", pass);
      if(!pass) return -1;
#ifdef DEBUG
      printf("After initializeSample...\n");
      cellData_gpu.print_sample();
#endif
    }

    // Sample particle information to cells
    nsample++ ;
    sampleParticles_gpu<<<particles.num_valid_particles/thrds_per_block+1, thrds_per_block>>>(
      cellData_gpu.raw_pointers,particles.raw_pointers) ;
    cudaDeviceSynchronize();
    cudaErrChk(cudaGetLastError(), "sampleParticles_gpu", pass);
    if(!pass) return -1;
#ifdef DEBUG
    printf("After sampleParticles...\n");
    cellData_gpu.print_sample();
#endif

    particle_count_map mapping(ni*nj*nk);
    mapping.map_particles_to_cells(particles);
#ifdef DEBUG
    printf("After sort by index...\n"); // inside map()
    particles.print_sample(2);

    printf("After particle count mapping...\n");
    mapping.print_size();
    mapping.print_sample();      
#endif

    // Compute particle collisions
    collideParticles_gpu<<<mapping.num_occupied_cells/thrds_per_block + 1,thrds_per_block>>>(
          particles.raw_pointers,collisionData_gpu.raw_pointers,
          cellData_gpu.raw_pointers,nsample,cellvol,sigmak,
          deltaT,pnum,mapping.raw_pointers,rand4State_ptr,
          randState_5_ptr,randState_6_ptr) ;
    cudaDeviceSynchronize();
    cudaErrChk(cudaGetLastError(), "collideParticles_gpu", pass);
    if(!pass) return -1;    
#ifdef DEBUG
      printf("After collideParticles...\n");
      particles.print_sample(4);      
#endif

    // print out progress
    if((n&0xf) == 0) {
      cout << n << ' ' << particles.num_valid_particles << endl ;
    }
    // particles = particleVec;
  }

  particleVec = particles.to_vector() ;
  
  // Write out final particle data
  ofstream ofile("particles.dat",ios::out) ;
  
  for(particle p : particleVec) {
    ofile << p.pos.x << ' ' << p.pos.y << ' ' <<  p.pos.z << ' ' <<  p.type <<  p.index << endl ;
  }

  ofile.close() ;
  
  cellData = cellData_gpu.to_vector();

  // Write out cell sampled data
  ofstream ocfile("cells.dat", ios::out) ;

  for(size_t i=0;i<cellData.size();++i)
    ocfile << cellData[i].nparticles << ' '
           << cellData[i].vel.x << ' '
           << cellData[i].vel.y << ' '
           << cellData[i].vel.z << ' '
           << cellData[i].energy << endl ;
  ocfile.close() ;

  return 0 ;
}
