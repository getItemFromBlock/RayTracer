#pragma once
#include "VectorDouble.cuh"
class Ray
{
public:
	__device__ Ray();
	__device__ Ray(VectorDouble originV, VectorDouble directionV);
	__device__ ~Ray();

	__device__ VectorDouble getOrigin();
	__device__ VectorDouble getDirection();
	__device__ VectorDouble at(double t);

	VectorDouble origin, direction;
};