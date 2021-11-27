#include <iostream>
#include <fstream>
#include <cmath>
#include <list>
#include <vector>
#include <stdlib.h>
#include "vect3d.h"
#include "structs.h"
#include <vector_types.h>

#include "helper_gl.h"
#include <GL/freeglut.h>

#include <chrono>
#include <thread>

using namespace std ;

// Physical constant describing atom collision size
const float sigmak = 1e-28 ; // collision cross section

// Note, pnum recomputed from mean particle per cell and density
float pnum = 1e27 ; // number of particles per simulated particle

// For this example, the geometry is a simple plate
float plate_x = -0.25 ;
float plate_dy = 0.25 ;
float plate_dz = .5 ;


uint ni=32,nj=32,nk=32 ;
// mean velocity and temperature of flow
float Mach = 20 ;
float vmean=1 ;
// mean particles per cell
int mppc = 10 ;
float density = 1e30 ; // Number of molecules per unit cube of space
float sim_time = -1 ;

double cellvol;
float vtemp;
int sample_reset = 0 ;
int nsample = 0 ;
int n = 0;
int ntimesteps = 0;

list<particle> particleList;
vector<cellSample> cellData;
vector<collisionInfo> collisionData;
float deltaT;


#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

// animation 
float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;
bool pause = false;
uint delay_ms = 0;

GLuint edges_buffer;
GLuint particles_buffer;
uint num_edges;
uint num_particles;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void catchKey(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

void save_output_data();

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

// Create particles at inflow boundary
// This works by creating particles at a ghost cell just before the boundary
// any particles that don't make it into the domain are discarded.
void initializeBoundaries(list<particle> &particleList,
                          int ni, int nj, int nk,
                          float vmean, float vtemp,int mppc) {
  double dx=2./float(ni),dy=2./float(nj),dz=2./float(nk) ;
  
  for(int j=0;j<nj;++j) 
    for(int k=0;k<nk;++k) {
      double cx = -1-dx ;
      double cy = -1+float(j)*dy ;
      double cz = -1+float(k)*dz ;
      for(int m=0;m<mppc;++m) { // Compute mcp particles
        particle p ;
        // random location within cell
        p.pos = vect3d(cx+ranf()*dx,cy+ranf()*dy,cz+ranf()*dz) ;
        // random velocity with given thermal velocity, vtemp
        p.vel = randomDir()*randVel(vtemp) ;
        p.vel.x += vmean ;
        p.type = 0 ;
        p.index = 0 ;
        particleList.push_back(p) ;
      }
    }
}


// Move particle for the timestep.  Also handle side periodic boundary
// conditions and specular reflections off of a plate 
void moveParticlesWithBCs(list<particle> &particleList, float deltaT) {
  list<particle>::iterator ii ;

  // Loop over particles
  for(ii=particleList.begin();ii!=particleList.end();++ii) {
    // position before and after timestep
    vect3d pos = ii->pos ;
    vect3d npos = pos+ii->vel*deltaT ;

    // Check if particle hits the plate
    if((pos.x < plate_x && npos.x > plate_x) ||
       (pos.x > plate_x && npos.x < plate_x)) {
      // It passes through the plane of the plate, now
      // check if it actually hits the plate
      double t = (pos.x-plate_x)/(pos.x-npos.x) ;
      vect3d pt = pos*(1.-t)+npos*t ;
      if((pt.y < plate_dy && pt.y > -plate_dy) &&
         (pt.z < plate_dz && pt.z > -plate_dz)) {
           // collides with plate
           // adjust position and velocity (specular reflection)
           npos.x = npos.x - 2*(npos.x-plate_x) ;
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
void removeOutsideParticles(list<particle> &particleList) {
  list<particle>::iterator ii,iin ;
  
  for(ii=particleList.begin();ii!=particleList.end();) {
    iin = ii ;
    iin++ ;
    if(ii->pos.x < -1 || ii->pos.x > 1) { // Outside domain so remove
      particleList.erase(ii) ;
    }
    ii = iin ;
  }
}


// Compute the cell index of each particle, for a regular Cartesian grid
// this can be computed directly from the coordinates
void indexParticles(list<particle> &particleList, int ni, int nj, int nk) {
  double dx=2./float(ni),dy=2./float(nj),dz=2./float(nk) ;
  list<particle>::iterator ii,iin ;
  
  for(ii=particleList.begin();ii!=particleList.end();++ii) {
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
                     const list<particle> &particleList) {
  list<particle>::const_iterator ii ;
  for(ii=particleList.begin();ii!=particleList.end();++ii) {
    int i = ii->index ;
    cellData[i].nparticles++ ;
    cellData[i].vel += ii->vel ;
    cellData[i].energy += .5*dot(ii->vel,ii->vel) ;
  }
} 
  

// Compute particle collisions
void collideParticles(list<particle> &particleList,
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
  list<particle>::iterator ii ;
  for(ii=particleList.begin();ii!=particleList.end();++ii) {
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
  for(ii=particleList.begin();ii!=particleList.end();++ii) {
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

void takeStep(
              // list<particle> &particleList, 
              // vector<cellSample> &cellData,
              // vector<collisionInfo> &collisionData,
              // float deltaT
              ) {
    // Add particles at inflow boundaries
    initializeBoundaries(particleList,ni,nj,nk,vmean,vtemp,mppc) ;


    // Move particles
    moveParticlesWithBCs(particleList,deltaT) ;
    // Remove any particles that are now outside of boundaries
    removeOutsideParticles(particleList) ;
    // Compute cell index for particles based on their current
    // locations
    indexParticles(particleList,ni,nj,nk) ;
    // If time to reset cell samples, reinitialize data
    if(n%sample_reset == 0 ) {
      initializeSample(cellData) ;
      nsample = 0 ;
    }
    // Sample particle information to cells
    nsample++ ;
    sampleParticles(cellData,particleList) ;
    // Compute particle collisions
    collideParticles(particleList,collisionData,cellData,nsample,
                     cellvol,deltaT) ;
    // print out progress
    if((n&0xf) == 0) {
      cout << n << ' ' << particleList.size() << endl ;
    }

    n++;
    if(n >= ntimesteps) save_output_data();
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
#if 0
  int ni=32,nj=32,nk=32 ;
  // mean velocity and temperature of flow
  float Mach = 20 ;
  float vmean=1 ;
  // mean particles per cell
  int mppc = 10 ;
  float density = 1e30 ; // Number of molecules per unit cube of space
  float time = -1 ;
#endif

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
      sim_time = atof(av[++i]) ;
    ++i ;
  }

  vtemp= vmean/Mach ;
  float dx=2./float(ni),dy=2./float(nj),dz=2./float(nk) ;
  cellvol = dx*dy*dz ;

  // Compute number of molecules a particle represents
  pnum = density*cellvol/float(mppc) ;

  // Create simulation data structures 
  // list<particle> particleList ;
  // vector<cellSample> 
  cellData.resize(ni*nj*nk) ;
  // vector<collisionInfo> 
  collisionData.resize(ni*nj*nk) ;

  // Compute reasonable timestep
  float deltax = 2./float(max(max(ni,nj),nk)) ;
  deltaT = .1*deltax/(vmean+vtemp) ;

  // If time duration not given, simulate for 4 free-stream flow-through times 
  if(sim_time < 0)
    sim_time = 8./(vmean+vtemp) ;

  // Compute nearest power of 2 timesteps
  float tsteps = sim_time/deltaT ;
  int ln2steps = int(ceil(log(tsteps)/log(2.0))) ;
  ntimesteps = 1<<ln2steps ;
  cout << "time = " << sim_time << ' ' << ", nsteps = " << ntimesteps << endl ;
  // int nsample = 0 ;

  // re-sample 4 times during simulation
  sample_reset = ntimesteps/4 ;

  // Begin simulation.  Initialize collision data
  initializeCollision(collisionData,vtemp) ;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

  initGL(&ac, av);
  glutCloseFunc(cleanup);
  glutMainLoop();


#if 0
  // Step forward in time
  for(int n=0;n<ntimesteps;++n) {
    // Add particles at inflow boundaries
    initializeBoundaries(particleList,ni,nj,nk,vmean,vtemp,mppc) ;
    // Move particles
    moveParticlesWithBCs(particleList,deltaT) ;
    // Remove any particles that are now outside of boundaries
    removeOutsideParticles(particleList) ;
    // Compute cell index for particles based on their current
    // locations
    indexParticles(particleList,ni,nj,nk) ;
    // If time to reset cell samples, reinitialize data
    if(n%sample_reset == 0 ) {
      initializeSample(cellData) ;
      nsample = 0 ;
    }
    // Sample particle information to cells
    nsample++ ;
    sampleParticles(cellData,particleList) ;
    // Compute particle collisions
    collideParticles(particleList,collisionData,cellData,nsample,
                     cellvol,deltaT) ;
    // print out progress
    if((n&0xf) == 0) {
      cout << n << ' ' << particleList.size() << endl ;
    }
#endif 
  
}

void save_output_data(){
  // Write out final particle data
  ofstream ofile("particles.dat",ios::out) ;
  list<particle>::iterator ii ;

  for(ii=particleList.begin();ii!=particleList.end();++ii) {
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

}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("DSMC - Sequential");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(catchKey);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }
    printf("openGL version = %s\n", glGetString(GL_VERSION));


    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);
    
    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 20.0);


    float cell_size = 2.0;
    // uint num_cells_on_edge = 4;
    // uint ni=num_cells_on_edge , nj=num_cells_on_edge , nk=num_cells_on_edge ;
    std::vector<float3> cells_edges_positions_data;

    float dx = cell_size/ni, dy = cell_size/nj, dz = cell_size/nk;
    float x, y, z;
    // Front, back (front = -1, back = 1)
    float backx = cell_size/2.0, frontx = -backx;
    for(uint j=0; j<nj+1; ++j) {
        y = -cell_size/2.0 + dy*j;
        for(uint k=0; k<nk+1; ++k) {
            z = -cell_size/2.0 + dz*k;
            cells_edges_positions_data.push_back({frontx,y,z});
            cells_edges_positions_data.push_back({backx,y,z});
        }
    }

    // Bottom, top (bottom = -1, top = 1)
    float topy = cell_size/2.0, bottomy = -topy;
    for(uint i=0; i<ni+1; ++i) {
        x = -cell_size/2.0 + dx*i;
        for(uint k=0; k<nk+1; ++k) {
            z = -cell_size/2.0 + dz*k;
            cells_edges_positions_data.push_back({x,bottomy,z});
            cells_edges_positions_data.push_back({x,topy,z});
        }
    }
    
    // Left, right (left = -1, right = 1)
    float rightz = cell_size/2.0, leftz = -rightz;
    for(uint i=0; i<ni+1; ++i) {
        x = -cell_size/2.0 + dx*i;
        for(uint j=0; j<nj+1; ++j) {
            y = -cell_size/2.0 + dy*j;
            cells_edges_positions_data.push_back({x,y,leftz});
            cells_edges_positions_data.push_back({x,y,rightz});
        }
    }

    num_edges  = (ni+1)*(nj+1) + (ni+1)*(nk+1) + (nj+1)*(nk+1);
    // printf("Number of edges = %u\n", num_edges);

#if 0
    for(uint i=0; i < num_edges; i+=num_edges/12){
        std::cout << 
        cells_edges_positions_data[i*2].x <<" "<< 
        cells_edges_positions_data[i*2].y <<" "<< 
        cells_edges_positions_data[i*2].z << 
        " | "<< 
        cells_edges_positions_data[i*2+1].x <<" "<< 
        cells_edges_positions_data[i*2+1].y <<" "<< 
        cells_edges_positions_data[i*2+1].z << 
        std::endl;
    }
#endif

    glGenBuffers(1, &edges_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, edges_buffer);
    glBufferData(GL_ARRAY_BUFFER, num_edges*2*sizeof(float3), cells_edges_positions_data.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glGenBuffers(1, &particles_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return true;
}

void copy_particles_to_buffer() {
    vector<float3> particles_pos_vec;
    list<particle>::iterator ii; 
    
    // This is more than likely quite slow....
    // std::vector<float3> particles_pos; 
    for(ii = particleList.begin(); ii != particleList.end(); ii++) {
        vect3d pos = ii->pos;
        particles_pos_vec.push_back({pos.x, pos.y, pos.z});
    }
    num_particles = particles_pos_vec.size();
    glBindBuffer(GL_ARRAY_BUFFER, particles_buffer);
    glBufferData(GL_ARRAY_BUFFER, num_particles*sizeof(float3), particles_pos_vec.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display(){
    
    // timestep here?
    if( (n<ntimesteps) & !pause) {
      takeStep();
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 0.5, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    glEnableClientState(GL_VERTEX_ARRAY);

    // glGenBuffers(1, &particles_buffer);
    copy_particles_to_buffer();
    glBindBuffer(GL_ARRAY_BUFFER, particles_buffer);
    glVertexPointer(3, GL_FLOAT, 0, (GLuint*) 0);

    
    glDrawArrays(GL_POINTS, 0, num_particles);

#if 0
    // Cube edges?
    // glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, edges_buffer);
    glVertexPointer(3, GL_FLOAT, 0, (GLuint*) 0);

    glDrawArrays(GL_LINES, 0, num_edges*2);
#endif 
    glBegin(GL_LINE_LOOP);
      glVertex3d(plate_x, -plate_dy, -plate_dz);
      glVertex3d(plate_x, -plate_dy, plate_dz);
      glVertex3d(plate_x, plate_dy, plate_dz);
      glVertex3d(plate_x, plate_dy, -plate_dz);
    glEnd();
    // glBindBuffer(GL_ARRAY_BUFFER, particles_positions_buffer);


    glutSwapBuffers();

    g_fAnim += 0.01f;

}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    if (edges_buffer){
        glBindBuffer(1, edges_buffer);
        glDeleteBuffers(1, &edges_buffer);
    }
    if (particles_buffer){
        glBindBuffer(1, particles_buffer);
        glDeleteBuffers(1, &particles_buffer);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) : // ESC
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
        case (32) : // space
                pause = !pause;
                return;
    }
}
void catchKey(int key, int x, int y)
{
  switch (key)
  {
    case (GLUT_KEY_LEFT) :
      if(delay_ms < 2000) delay_ms += 100;
      return;
    case (GLUT_KEY_RIGHT) :
      if(delay_ms > 0) delay_ms -= 100;
      if(delay_ms > 2500) delay_ms = 0;
      return;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
