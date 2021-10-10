#pragma once
#include "../tools/HitRecord.cuh"
#include "Drawable.cuh"

class VoidDrawable :public  Drawable
{
public:
	__device__ VoidDrawable();
	__device__ VoidDrawable(const Drawable& obj);
	__device__ ~VoidDrawable();

	__device__ virtual HitRecord Drawable::hit(Ray r, float tmin, float tmax) override;
	__device__ virtual Color6Component Drawable::getColor(HitRecord* hit) override;
	__device__ virtual bool Drawable::doReflect() override;
	__device__ virtual int Drawable::getSubType() override;
private:
};