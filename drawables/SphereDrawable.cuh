#pragma once
#include "HitRecord.cuh"
#include "Drawable.cuh"
class SphereDrawable :public Drawable
{
public:
	__device__ SphereDrawable(VectorDouble, double, Color6Component);
	__device__ SphereDrawable();
	__device__ SphereDrawable(const Drawable& obj);
	__device__ ~SphereDrawable();

	__device__ virtual HitRecord Drawable::hit(Ray r, double tmin, double tmax) override;
	__device__ virtual Color6Component Drawable::getColor(HitRecord* hit) override;
	__device__ virtual double Drawable::getVHitBox(VectorDouble* position) override;
	__device__ virtual bool Drawable::doReflect() override;
private:
	VectorDouble pos;
	const double radius;
	const Color6Component color;
};