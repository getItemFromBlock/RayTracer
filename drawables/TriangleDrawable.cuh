#pragma once
#include "HitRecord.cuh"
#include "Drawable.cuh"
class TriangleDrawable :public Drawable
{
public:
	__device__ TriangleDrawable(VectorDouble, VectorDouble, VectorDouble, Color6Component);
	__device__ TriangleDrawable();
	__device__ TriangleDrawable(const Drawable& obj);
	__device__ ~TriangleDrawable();

	__device__ virtual HitRecord Drawable::hit(Ray r, double tmin, double tmax) override;
	__device__ virtual Color6Component Drawable::getColor(HitRecord* hit) override;
	__device__ virtual double Drawable::getVHitBox(VectorDouble* position) override;
	__device__ virtual bool Drawable::doReflect() override;
private:
	VectorDouble A, B, C, normal, AB, AC;
	const Color6Component color;
};