#pragma once
#include "../tools/HitRecord.cuh"
#include "Drawable.cuh"
class TriangleMirrorDrawable :public Drawable
{
public:
	__device__ TriangleMirrorDrawable(Vector3D, Vector3D, Vector3D, Color6Component,int);
	__device__ TriangleMirrorDrawable();
	__device__ TriangleMirrorDrawable(const Drawable& obj);
	__device__ ~TriangleMirrorDrawable();

	__device__ virtual HitRecord Drawable::hit(Ray r, float tmin, float tmax) override;
	__device__ virtual Color6Component Drawable::getColor(HitRecord* hit) override;
	__device__ virtual bool Drawable::doReflect() override;
	__device__ virtual int Drawable::getSubType() override;
private:
	Vector3D A, B, C, normal, AB, AC;
	const Color6Component color;
	const int type;
};