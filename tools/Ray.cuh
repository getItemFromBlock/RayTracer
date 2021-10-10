#pragma once
#include "Vector3D.cuh"
class Ray
{
public:
	__host__ __device__ Ray();
	__host__ __device__ Ray(Vector3D originV, Vector3D directionV);
	__host__ __device__ ~Ray();

	__host__ __device__ Vector3D getOrigin();
	__host__ __device__ Vector3D getDirection();
	__host__ __device__ Vector3D at(float t);

	Vector3D origin, direction;
};