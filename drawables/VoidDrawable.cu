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

__device__ HitRecord VoidDrawable::hit(Ray r, double tmin, double tmax)
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

__device__ double VoidDrawable::getVHitBox(VectorDouble* position)
{
	return -50000;
}