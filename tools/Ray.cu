#include "Ray.cuh"

__host__ __device__ Ray::Ray(Vector3D originV, Vector3D directionV) : origin(originV), direction(directionV) // constructeur - définition
{

}

__host__ __device__ Ray::Ray() : origin(Vector3D()), direction(Vector3D())
{

}

__host__ __device__ Ray::~Ray() // Destructeur - définition
{

}

__host__ __device__ Vector3D Ray::getOrigin()
{
	return origin;
}

__host__ __device__ Vector3D Ray::getDirection()
{
	return direction;
}

__host__ __device__ Vector3D Ray::at(float t)
{
	return origin.add(direction.mul(t));
}