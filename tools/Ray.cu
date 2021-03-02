#include "Ray.cuh"

__device__ Ray::Ray(VectorDouble originV, VectorDouble directionV) : origin(originV), direction(directionV) // constructeur - définition
{

}

__device__ Ray::Ray() : origin(VectorDouble()), direction(VectorDouble())
{

}

__device__ Ray::~Ray() // Destructeur - définition
{

}

__device__ VectorDouble Ray::getOrigin()
{
	return origin;
}

__device__ VectorDouble Ray::getDirection()
{
	return direction;
}

__device__ VectorDouble Ray::at(double t)
{
	return origin.add(direction.mul(t));
}