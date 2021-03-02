#include "HitRecord.cuh"

__device__ HitRecord::HitRecord(VectorDouble pointT, VectorDouble normalT, double T) : point(pointT), normal(normalT), t(T) // constructeur - définition
{
	isEmpty = false;
}

__device__ HitRecord::HitRecord() : point(VectorDouble()), normal(VectorDouble()), t(100000), isEmpty(true)
{
	isEmpty = true;
}

__device__ HitRecord::~HitRecord() // Destructeur - définition
{

}

__device__ void HitRecord::setFaceNormal(Ray r, VectorDouble outward_normal)
{
	front_face = r.getDirection().dot(outward_normal) < 0;
	normal = front_face ? outward_normal : outward_normal.mul(-1.0);
}