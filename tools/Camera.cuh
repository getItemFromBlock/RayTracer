#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include "Ray.cuh"

class Camera
{
public:
	__device__ Camera(Vector3D lookfrom, Vector3D direction, Vector3D vup, float vfov, float aspect_ratio);	// Constructeur - déclaration
	__device__ Camera(); // Compat
	__device__ ~Camera(); // Destructeur

	__device__ void refresh(Vector3D lookfrom, Vector3D direction, Vector3D vup, float vfov, float aspect_ratio);

	__device__ Ray getRay(float u, float v);

	__device__ float toRadians(float input);
private:
	Vector3D origin;
	Vector3D horizontal;
	Vector3D vertical;
	Vector3D lower_left_corner;

};