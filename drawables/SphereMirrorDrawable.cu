#include "SphereMirrorDrawable.cuh"


__device__ SphereMirrorDrawable::SphereMirrorDrawable(Vector3D position, float rayon, Color6Component couleur, int subType) : // constructeur - définition
	pos(position), radius(rayon), color(couleur), type(subType)
{
}

__device__ SphereMirrorDrawable::SphereMirrorDrawable() :
	pos(Vector3D()), radius(1), color(Color6Component()), type(0)
{
}

__device__ SphereMirrorDrawable::SphereMirrorDrawable(const Drawable& obj) : Drawable(obj), radius(1), type(0)
{

}

__device__ SphereMirrorDrawable::~SphereMirrorDrawable() // Destructeur - définition
{

}

__device__ HitRecord SphereMirrorDrawable::hit(Ray r, float tmin, float tmax)
{
	HitRecord rec = HitRecord();
	rec.isEmpty = true;
	Vector3D oc = (r).getOrigin().sub(pos);
	float a = (r).getDirection().length_squared();
	float half_b = oc.dot((r).getDirection());
	float c = oc.length_squared() - radius * radius;
	float discriminant = half_b * half_b - a * c;
	if (discriminant > 0)
	{
		float root = sqrt(discriminant);
		float temp = (-half_b - root) / a;
		if (temp < tmax && temp > tmin)
		{
			rec.t = temp;
			rec.point = (r).at(rec.t);
			Vector3D outward_normal = (rec.point.sub(pos)).div(radius);
			rec.setFaceNormal((r), outward_normal);
			if (rec.front_face) {
				rec.isEmpty = false;
				rec.normal = r.getDirection().sub(rec.normal.unitVector().mul((r.getDirection().dot(rec.normal.unitVector()) * 2.0)));
				return rec;
			}
		}
		temp = (-half_b + root) / a;
		if (temp <= tmax && temp > tmin)
		{
			rec.t = temp;
			rec.point = (r).at(rec.t);
			Vector3D outward_normal = (rec.point.sub(pos)).div(radius);
			rec.setFaceNormal((r), outward_normal);
			if (rec.front_face) {
				rec.isEmpty = false;
				rec.normal = r.getDirection().sub(rec.normal.unitVector().mul((r.getDirection().dot(rec.normal.unitVector()) * 2.0)));
				return rec;
			}
		}
	}
	return rec;
}

__device__ Color6Component SphereMirrorDrawable::getColor(HitRecord* hit)
{
	return color;
}

__device__ bool SphereMirrorDrawable::doReflect()
{
	return true;
}

__device__ int SphereMirrorDrawable::getSubType() {
	return type;
}