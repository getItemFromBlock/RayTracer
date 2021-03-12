#pragma once
#include "HitRecord.cuh"
#include "Color6Component.cuh"

class Drawable
{
public:
	__device__ virtual HitRecord hit(Ray r, double tmin, double tmax) = 0;
	__device__ virtual Color6Component getColor(HitRecord* hit) = 0;
	__device__ virtual double getVHitBox(VectorDouble* pos) = 0;
	__device__ virtual bool doReflect() = 0;
};