#ifndef VECT3D_H
#define VECT3D_H
#include <cmath>
#include <iostream>

//---------------------vect3d------------------//
struct vect3d {
  float x,y,z ;
  __host__ __device__ vect3d() {} 
  __host__ __device__ vect3d(float xx,float yy, float zz) : x(xx),y(yy),z(zz) {}
  __host__ __device__ vect3d(const vect3d &v) {x=v.x;y=v.y;z=v.z;}
} ;
  
inline std::ostream & operator<<(std::ostream &s, const vect3d &v) {
  s << v.x << ' ' << v.y << ' ' << v.z << ' ' ;
  return s ;
}

inline std::istream &operator>>(std::istream &s, vect3d &v) {
  s >> v.x >> v.y >> v.z ;
  return s ;
}

__host__ __device__
inline float dot(const vect3d &v1, const vect3d &v2) {
  return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z ;
}

__host__ __device__
inline float norm(const vect3d &v) {
  return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z) ;
}

__host__ __device__
inline vect3d cross(const vect3d &v1, const vect3d &v2) {
  return vect3d(v1.y*v2.z-v1.z*v2.y,
                v1.z*v2.x-v1.x*v2.z,
                v1.x*v2.y-v1.y*v2.x) ;
}

__host__ __device__
inline vect3d &operator*=(vect3d &target, float val) {
  target.x *= val ;
  target.y *= val ;
  target.z *= val ;
  return target ;
}

__host__ __device__
inline vect3d &operator/=(vect3d &target, float val) {
  target.x /= val ;
  target.y /= val ;
  target.z /= val ;
  return target ;
}

__host__ __device__
inline vect3d &operator*=(vect3d &target, double val) {
  target.x *= val ;
  target.y *= val ;
  target.z *= val ;
  return target ;
}

__host__ __device__
inline vect3d &operator/=(vect3d &target, double val) {
  target.x /= val ;
  target.y /= val ;
  target.z /= val ;
  return target ;
}

__host__ __device__
inline vect3d &operator*=(vect3d &target, long double val) {
  target.x *= val ;
  target.y *= val ;
  target.z *= val ;
  return target ;
}

__host__ __device__
inline vect3d &operator/=(vect3d &target, long double val) {
  target.x /= val ;
  target.y /= val ;
  target.z /= val ;
  return target ;
}

__host__ __device__
inline vect3d operator+=(vect3d &target, const vect3d &val) {
  target.x += val.x ;
  target.y += val.y ;
  target.z += val.z ;
  return target ;
}

__host__ __device__
inline vect3d operator-=(vect3d &target, const vect3d &val) {
  target.x -= val.x ;
  target.y -= val.y ;
  target.z -= val.z ;
  return target ;
}

__host__ __device__
inline vect3d operator+(const vect3d &v1, const vect3d &v2) {
  return vect3d(v1.x+v2.x,v1.y+v2.y,v1.z+v2.z) ;
}

__host__ __device__
vect3d operator-(const vect3d &v1, const vect3d &v2) {
    return vect3d(v1.x-v2.x,v1.y-v2.y,v1.z-v2.z) ;
  }

__host__ __device__
inline vect3d operator*(const vect3d &v1, float r2) {
    return vect3d(v1.x*r2,v1.y*r2,v1.z*r2) ;
  }

__host__ __device__
inline vect3d operator*(float r1, const vect3d &v2) {
    return vect3d(v2.x*r1,v2.y*r1,v2.z*r1) ;
  }

__host__ __device__
inline vect3d operator/(const vect3d &v1, float r2) {
    return vect3d(v1.x/r2,v1.y/r2,v1.z/r2) ;
  }

__host__ __device__
inline vect3d operator*(const vect3d &v1, double r2) {
    return vect3d(v1.x*r2,v1.y*r2,v1.z*r2) ;
  }

__host__ __device__
inline vect3d operator*(double r1, const vect3d &v2) {
    return vect3d(v2.x*r1,v2.y*r1,v2.z*r1) ;
  }

__host__ __device__
inline vect3d operator/(const vect3d &v1, double r2) {
    return vect3d(v1.x/r2,v1.y/r2,v1.z/r2) ;
  }

__host__ __device__
inline vect3d operator*(const vect3d &v1, long double r2) {
    return vect3d(v1.x*r2,v1.y*r2,v1.z*r2) ;
  }

__host__ __device__
inline vect3d operator*(long double r1, const vect3d &v2) {
    return vect3d(v2.x*r1,v2.y*r1,v2.z*r1) ;
  }

__host__ __device__
inline vect3d operator/(const vect3d &v1, long double r2) {
    return vect3d(v1.x/r2,v1.y/r2,v1.z/r2) ;
}
#endif
