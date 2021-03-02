#pragma once
#include <cuda_runtime.h>
#ifndef __CUDA_ARCH__
#include <math.h>
#endif


class VectorDouble
{
public:
	__host__ __device__ VectorDouble(double, double, double);	// Constructeur - déclaration
	__host__ __device__ VectorDouble();
	__host__ __device__ ~VectorDouble();		// Destructeur - déclaration

	__host__ __device__ VectorDouble operator=(const VectorDouble other); //Copy assignement

	__host__ __device__ void setX(double arg);
	__host__ __device__ void setY(double arg);
	__host__ __device__ void setZ(double arg);

	__host__ __device__ double getX();
	__host__ __device__ double getY();
	__host__ __device__ double getZ();

	__host__ __device__ double getLength();
	__host__ __device__ VectorDouble add(VectorDouble);
	__host__ __device__ VectorDouble sub(VectorDouble);
	__host__ __device__ VectorDouble mul(VectorDouble);
	__host__ __device__ VectorDouble mul(double d);
	__host__ __device__ VectorDouble div(double d);

	__host__ __device__ double dot(VectorDouble v);
	__host__ __device__ VectorDouble cross(VectorDouble v);
	__host__ __device__ VectorDouble unitVector();
	__host__ __device__ double length_squared();
private:
	double X, Y, Z;
};