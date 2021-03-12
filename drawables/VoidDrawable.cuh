#pragma once
#include "Drawable.cuh"

class VoidDrawable :public  Drawable
{
public:
	__device__ VoidDrawable();
	__device__ VoidDrawable(const Drawable& obj);
	__device__ ~VoidDrawable();

	__device__ virtual HitRecord Drawable::hit(Ray r, double tmin, double tmax) override;
	__device__ virtual Color6Component Drawable::getColor(HitRecord* hit) override;
	__device__ virtual double Drawable::getVHitBox(VectorDouble* position) override;
	__device__ virtual bool Drawable::doReflect() override;
private:
};