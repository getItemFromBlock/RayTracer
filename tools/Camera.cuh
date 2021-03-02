#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include "Ray.cuh"

class Camera
{
public:
	__device__ Camera(VectorDouble lookfrom, VectorDouble direction, VectorDouble vup, double vfov, double aspect_ratio);	// Constructeur - déclaration
	__device__ Camera(); // Compat
	__device__ ~Camera(); // Destructeur

	__device__ void refresh(VectorDouble lookfrom, VectorDouble direction, VectorDouble vup, double vfov, double aspect_ratio);

	__device__ Ray getRay(double u, double v);

	__device__ double toRadians(double input);
private:
	VectorDouble origin;
	VectorDouble horizontal;
	VectorDouble vertical;
	VectorDouble lower_left_corner;

};