#pragma once
#include "VectorDouble.cuh"
#include "Ray.cuh"
class HitRecord
{
public:
	__device__ HitRecord();
	__device__ HitRecord(VectorDouble pointT, VectorDouble normalT, double T);
	__device__ ~HitRecord();

	__device__ void setFaceNormal(Ray r, VectorDouble outward_normal);

	VectorDouble point, normal;
	double t, u, v;
	bool front_face;
	bool isEmpty;
};