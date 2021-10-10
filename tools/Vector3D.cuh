#pragma once
#include <cuda_runtime.h>
#ifndef __CUDA_ARCH__
#include <math.h>
#endif


class Vector3D
{
public:
	__host__ __device__ Vector3D(float, float, float);	// Constructeur - déclaration
	__host__ __device__ Vector3D();
	__host__ __device__ ~Vector3D();		// Destructeur - déclaration

	__host__ __device__ Vector3D operator=(const Vector3D other); //Copy assignement

	__host__ __device__ void setX(float arg);
	__host__ __device__ void setY(float arg);
	__host__ __device__ void setZ(float arg);

	__host__ __device__ float getX();
	__host__ __device__ float getY();
	__host__ __device__ float getZ();

	__host__ __device__ float getLength();
	__host__ __device__ Vector3D add(Vector3D);
	__host__ __device__ Vector3D sub(Vector3D);
	__host__ __device__ Vector3D mul(Vector3D);
	__host__ __device__ Vector3D mul(float d);
	__host__ __device__ Vector3D div(float d);

	__host__ __device__ float dot(Vector3D v);
	__host__ __device__ Vector3D cross(Vector3D v);
	__host__ __device__ Vector3D unitVector();
	__host__ __device__ float length_squared();
private:
	float X, Y, Z;
};