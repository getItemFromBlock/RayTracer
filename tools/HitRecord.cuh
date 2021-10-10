#pragma once
#include "Vector3D.cuh"
#include "Ray.cuh"
class HitRecord
{
public:
	__host__ __device__ HitRecord();
	__host__ __device__ HitRecord(Vector3D pointT, Vector3D normalT, float T);
	__host__ __device__ ~HitRecord();

	__host__ __device__ void setFaceNormal(Ray r, Vector3D outward_normal);

	Vector3D point, normal;
	float t = 0.0, u = 0.0, v = 0.0;
	bool front_face = false;
	bool isEmpty = true;
};