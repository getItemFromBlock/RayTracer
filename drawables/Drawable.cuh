#pragma once
#include "../tools/HitRecord.cuh"
#include "../tools/Color6Component.cuh"

class Drawable
{
public:
	__device__ virtual HitRecord hit(Ray r, float tmin, float tmax) = 0;
	__device__ virtual Color6Component getColor(HitRecord* hit) = 0;
	__device__ virtual bool doReflect() = 0;
	__device__ virtual int getSubType() = 0;
};