#include "VoidDrawable.cuh"

__device__ VoidDrawable::VoidDrawable()
{
}

__device__ VoidDrawable::VoidDrawable(const Drawable& obj) : Drawable(obj)
{
}

__device__ VoidDrawable::~VoidDrawable()
{
}

__device__ HitRecord VoidDrawable::hit(Ray r, float tmin, float tmax)
{
	return HitRecord();
}

__device__ Color6Component VoidDrawable::getColor(HitRecord* hit)
{
	return Color6Component();
}

__device__ bool VoidDrawable::doReflect()
{
	return false;
}

__device__ int VoidDrawable::getSubType() {
	return 0;
}