#include <iostream>
#include <fstream>
using namespace std ;

// Extract particle data from the z=0 plane and categorize them into files
// for plotting.
int main() {
  ifstream pin("particles.dat",ios::in) ;
  ofstream p0("p0.dat",ios::out) ;
  ofstream p1("p1.dat",ios::out) ;
  ofstream p2("p2.dat",ios::out) ;

  while(!pin.fail()) {
    float x,y,z ;
    int t ;
    pin >> x >> y >> z >> t ;
    if(pin.fail() || pin.eof()) 
      break ;
    if(z > -.1 && z < .1) {
      if(t==0) 
        p0 << x << ' ' << y << endl ;
      if(t==1)
        p1 << x << ' ' << y << endl ;
      if(t==2)
        p2 << x << ' ' << y << endl ;
    }
  }
}

        
      
    
  
