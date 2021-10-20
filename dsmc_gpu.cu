#include <iostream>
#include <fstream>
#include <cmath>
// #include <list>
#include <vector>
#include <stdlib.h>

#include "vect3d.h"
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
inline vect3d randomDir() {
  double B = 2.*ranf()-1 ;
  double A = sqrt(1.-B*B) ;
  double theta = ranf()*2*M_PI ;
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
    particles.pz[idx*mppc + m] = cx + rand_results4.x*dx;

    double t = max(rand_results4.w, 1e-200);
    double speed = vtemp * sqrt(-log(t));

    double B = 2. * rand_result_5-1;
    double A = sqrt(1.-B*B);
    double theta = rand_result_6*2*M_PI;
    particles.vx[idx*mppc + m] = B;
    particles.vy[idx*mppc + m] = A * cos(theta);
    particles.vz[idx*mppc + m] = A * sin(theta);
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
void moveParticlesWithBCs(vector<particle> &particleVec, float deltaT) {
  vector<particle>::iterator ii ;

  // Loop over particles
  for(ii=particleVec.begin();ii!=particleVec.end();++ii) {
    if (ii->type == -1) continue;
    // position before and after timestep
    vect3d pos = ii->pos ;
    vect3d npos = pos+ii->vel*deltaT ;

    // Check if particle hits the plate
    if((pos.x < plate_x && npos.x > plate_x) ||
       (pos.x > plate_x && npos.x < plate_x)) {
      // It passes through the plane of the plate, now
      // check if it actually hits the plate
      double t = (pos.x-plate_x)/(pos.x-npos.x) ; // fraction of timestep to hit plate
      vect3d pt = pos*(1.-t)+npos*t ; // interpolated position at time t
      if((pt.y < plate_dy && pt.y > -plate_dy) &&
         (pt.z < plate_dz && pt.z > -plate_dz)) {
           // collides with plate
           // adjust position and velocity (specular reflection)
           npos.x = npos.x - 2*(npos.x-plate_x) ; 
           // why 2 and not some function of t? b/c is calculating new npos, not pos,
           //   so has to get back to plate, then keeps moving the same amount in new new direction
           ii->vel.x = -ii->vel.x ; // Velocity just reflects along x direction
           ii->type = 2 ;
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
    ii->pos = npos ;
  }
}

// After moving particles, any particles outside of the cells need to be
// discarded as they cannot be indexed.  Since this can happen only at the
// x=-1 or x=1 boundaries, we only need to check the x coordinate
void removeOutsideParticles(vector<particle> &particleVec) {
  vector<particle>::iterator ii ;
  
  for(ii=particleVec.begin();ii!=particleVec.end();++ii) {
    if (ii->type == -1) continue;
    // iin = ii ;
    // iin++ ;
    if(ii->pos.x < -1 || ii->pos.x > 1) { // Outside domain so remove
      ii->type= -1 ;
    }
    ii++;
    // ii = iin ;
  }
}


// Compute the cell index of each particle, for a regular Cartesian grid
// this can be computed directly from the coordinates
void indexParticles(vector<particle> &particleVec, int ni, int nj, int nk) {
  double dx=2./float(ni),dy=2./float(nj),dz=2./float(nk) ;
  vector<particle>::iterator ii ;
  
  for(ii=particleVec.begin();ii!=particleVec.end();++ii) {
    if (ii->type == -1) continue;
    // For a Cartesian grid, the mapping from cell to particle is trivial
    int i = min(int(floor((ii->pos.x+1.0)/dx)),ni-1) ;
    int j = min(int(floor((ii->pos.y+1.0)/dy)),nj-1) ;
    int k = min(int(floor((ii->pos.z+1.0)/dz)),nk-1) ;
    ii->index = i*nj*nk+j*nk+k ;
  }  
}

// Initialize the sampled cell variables to zero
void initializeSample(vector<cellSample> &cellSample) {
  for(size_t i=0;i<cellSample.size();++i) {
    cellSample[i].nparticles = 0 ;
    cellSample[i].vel = vect3d(0,0,0) ;
    cellSample[i].energy=0 ;
  }
}

// Sum particle information to cell samples, this will be used to compute
// collision probabilities
void sampleParticles(vector<cellSample> &cellData, 
                     const vector<particle> &particleVec) {
  vector<particle>::const_iterator ii ;
  for(ii=particleVec.begin();ii!=particleVec.end();++ii) {
    if (ii->type == -1) continue;
    int i = ii->index ;
    cellData[i].nparticles++ ;
    cellData[i].vel += ii->vel ;
    cellData[i].energy += .5*dot(ii->vel,ii->vel) ;
  }
} 
  

// Compute particle collisions
void collideParticles(vector<particle> &particleVec,
                      vector<collisionInfo> &collisionData,
                      const vector<cellSample> &cellData,
                      int nsample, float cellvol, float deltaT) {
  int ncells = cellData.size() ;

  // Compute number of particles per cell and compute a set of pointers
  // from each cell to the corresponding particles
  vector<int> np(ncells),cnt(ncells) ;
  for(int i=0;i<ncells;++i) {
    np[i] = 0 ;
    cnt[i] = 0 ;
  }
  vector<particle>::iterator ii ;
  for(ii=particleVec.begin();ii!=particleVec.end();++ii) {
    if (ii->type == -1) continue;
    int i = ii->index ;
    np[i]++ ;
  }
  // Offsets will contain the index in the pmap data structure where
  // the pointers to particles for the given cell will begin
  vector<int> offsets(ncells+1) ;
  offsets[0] = 0 ;
  for(int i=0;i<ncells;++i)
    offsets[i+1] = offsets[i]+np[i] ;
  // pmap is a structure of pointers from cells to particles, note
  // since there may be many particles per cell, the offsets need to
  // be used to access particles from this data structure.
  vector<particle *> pmap(offsets[ncells]) ;
  for(ii=particleVec.begin();ii!=particleVec.end();++ii) {
    if (ii->type == -1) continue;
    int i = ii->index ;
    pmap[cnt[i]+offsets[i]] = &(*ii) ;
    cnt[i]++ ;
  }
  // Loop over cells and select particles to perform collisions
  for(int i=0;i<ncells;++i) {
    // Compute mean and instantaneous particle numbers for the cell
    float n_mean = float(cellData[i].nparticles)/float(nsample) ;
    float n_instant = np[i] ;
    // Compute a number of particles that need to be selected for
    // collision tests
    float select = n_instant*n_mean*pnum*collisionData[i].maxCollisionRate*deltaT/cellvol + collisionData[i].collisionRemainder ;
    // We can only check an integer number of collisions in any timestep
    int nselect = int(select) ;
    // The remainder collision fraction is saved for next timestep
    collisionData[i].collisionRemainder = select - float(nselect) ;
    if(nselect > 0) { // selected particles for collision
      if(np[i] < 2) { // if not enough particles for collision, wait until
        // we have enough
        collisionData[i].collisionRemainder += nselect ;
      } else {
        // Select nselect particles for possible collision
        float cmax = collisionData[i].maxCollisionRate ;
        for(int c=0;c<nselect;++c) {
          // select two points in the cell
          int pt1 = min(int(floor(ranf()*n_instant)),np[i]-1) ;
          int pt2 = min(int(floor(ranf()*n_instant)),np[i]-1) ;

          // Make sure they are unique points
          while(pt1==pt2)
            pt2 = min(int(floor(ranf()*n_instant)),np[i]-1) ;

          // Compute the relative velocity of two particles
          vect3d v1 = pmap[offsets[i]+pt1]->vel ;
          vect3d v2 = pmap[offsets[i]+pt2]->vel ;
          vect3d vr = v1-v2 ;
          float vrm = norm(vr) ;
          // Compute collision  rate for hard sphere model
          float crate = sigmak*vrm ;
          if(crate > cmax)
            cmax = crate ;
          // Check if these particles actually collide
          if(ranf() < crate/collisionData[i].maxCollisionRate) {
            // Collision Accepted, adjust particle velocities
            // Compute center of mass velocity, vcm
            vect3d vcm = .5*(v1+v2) ;
            // Compute random perturbation that conserves momentum
            vect3d vp = randomDir()*vrm ;

            // Adjust particle velocities to reflect collision
            pmap[offsets[i]+pt1]->vel = vcm + 0.5*vp ;
            pmap[offsets[i]+pt2]->vel = vcm - 0.5*vp ;

            // Bookkeeping to track particle interactions
            int t1 = pmap[offsets[i]+pt1]->type ;
            int t2 = pmap[offsets[i]+pt2]->type ;
            int tc = (t1+t2>0)?1:0 ;
            pmap[offsets[i]+pt1]->type = max(tc,t1) ;
            pmap[offsets[i]+pt2]->type = max(tc,t2) ;
          }
        }
        // Update the maximum collision rate to be used in future timesteps
        // for determining number of particles to select.
        collisionData[i].maxCollisionRate = cmax ;
      }
    }
  }
}


// Initialize the collision data structures to modest initial values
void initializeCollision(vector<collisionInfo> &collisionData,
                         float vtemp) {
  const int ncells = collisionData.size() ;
  for(int i=0;i<ncells;++i) {
    collisionData[i].maxCollisionRate = sigmak*vtemp ;
    collisionData[i].collisionRemainder = ranf() ;
  }
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
      plate_dz = 1 ;
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
      plate_x = atof(av[++i]) ;
    if(opt == "-platedy")
      plate_dy = atof(av[++i]) ;
    if(opt == "-platedz")
      plate_dz = atof(av[++i]) ;
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
  vector<collisionInfo> collisionData(ni*nj*nk) ;

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
  curandState * rand4State_6_ptr = raw_pointer_cast(randState_6.data());
  init_rands<<<thrds_per_block, num_cells/thrds_per_block>>> (
    seed, rand4State_ptr, randState_5_ptr, rand4State_6_ptr);
    
  // Begin simulation.  Initialize collision data
  initializeCollision(collisionData,vtemp) ;

  bool pass = true;
#ifdef DEBUG
  particles.print_size();
#endif
  // Step forward in time
  for(int n=0;n<ntimesteps;++n) {
    // Add particles at inflow boundaries
    // initializeBoundaries(particleVec,ni,nj,nk,vmean,vtemp,mppc) ;

    initializeBoundaries_gpu<<<nj*nk/thrds_per_block, thrds_per_block>>>(
                            particles.empty_raw_pointers,ni,nj,nk,vmean,vtemp,mppc, 
                            rand4State_ptr, randState_5_ptr, rand4State_6_ptr) ;
    cudaDeviceSynchronize();
    cudaErrChk(cudaGetLastError(), "initializeBoundaries_gpu", pass);
    if(!pass) return -1;

    particleVec = particles.valid_particles();
#ifdef DEBUG
    for(int i=particles.num_valid_particles - 12; i<particles.num_valid_particles + 12; i+=6 ) {
      for(int j=0; j<6; ++j) {
        int idx = i + j;
        printf("%4d:(%1.5f,%1.5f,%1.5f) | ", idx, particles.h_pos_x[idx], particles.h_pos_y[idx], particles.h_pos_z[idx]);
      }
      printf("\n");
    }
    printf("\n");
#endif
    // Move particles
    moveParticlesWithBCs(particleVec,deltaT) ;
    // Remove any particles that are now outside of boundaries
    removeOutsideParticles(particleVec) ;
    // Compute cell index for particles based on their current
    // locations
    indexParticles(particleVec,ni,nj,nk) ;
    // If time to reset cell samples, reinitialize data
    if(n%sample_reset == 0 ) {
      initializeSample(cellData) ;
      nsample = 0 ;
    }
    // Sample particle information to cells
    nsample++ ;
    sampleParticles(cellData,particleVec) ;
    // Compute particle collisions
    collideParticles(particleVec,collisionData,cellData,nsample,
                     cellvol,deltaT) ;
    // print out progress
    if((n&0xf) == 0) {
      cout << n << ' ' << particleVec.size() << endl ;
    }
    particles = particleVec;
  }

  // Write out final particle data
  ofstream ofile("particles.dat",ios::out) ;
  vector<particle>::iterator ii ;

  for(ii=particleVec.begin();ii!=particleVec.end();++ii) {
    ofile << ii->pos.x << ' ' << ii->pos.y << ' ' << ii ->pos.z << ' ' << ii->type << endl ;
  }

  ofile.close() ;
  
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
