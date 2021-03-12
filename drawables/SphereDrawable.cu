#include "SphereDrawable.cuh"


__device__ SphereDrawable::SphereDrawable(VectorDouble position, double rayon, Color6Component couleur): // constructeur - définition
	pos(position), radius(rayon), color(couleur)
{
}

__device__ SphereDrawable::SphereDrawable():
	pos(VectorDouble()), radius(1), color(Color6Component())
{
}

__device__ SphereDrawable::SphereDrawable(const Drawable& obj) : Drawable(obj), radius(1)
{

}

__device__ SphereDrawable::~SphereDrawable() // Destructeur - définition
{

}

__device__ HitRecord SphereDrawable::hit(Ray r, double tmin, double tmax)
{
	HitRecord rec = HitRecord();
	rec.isEmpty = true;
	int test = r.getDirection().getX();
	VectorDouble oc = (r).getOrigin().sub(pos);
	double a = (r).getDirection().length_squared();
	double half_b = oc.dot((r).getDirection());
	double c = oc.length_squared() - radius * radius;
	double discriminant = half_b * half_b - a * c;
	if (discriminant > 0)
	{
		double root = sqrt(discriminant);
		double temp = (-half_b - root) / a;
		int test2 = 1000 * temp;
		if (temp < tmax && temp > tmin)
		{
			rec.t = temp;
			rec.point = (r).at(rec.t);
			VectorDouble outward_normal = (rec.point.sub(pos)).div(radius);
			rec.setFaceNormal((r), outward_normal);
			if (rec.front_face) {
				rec.isEmpty = false;
				return rec;
			}
		}
		temp = (-half_b + root) / a;
		int test3 = 1000 * temp;
		if (temp <= tmax && temp > tmin)
		{
			rec.t = temp;
			rec.point = (r).at(rec.t);
			VectorDouble outward_normal = (rec.point.sub(pos)).div(radius);
			rec.setFaceNormal((r), outward_normal);
			if (rec.front_face) {
				rec.isEmpty = false;
				return rec;
			}
		}
	}
	return rec;
}

__device__ Color6Component SphereDrawable::getColor(HitRecord* hit)
{
	return color;
}

__device__ bool SphereDrawable::doReflect()
{
	return false;
}

__device__ double SphereDrawable::getVHitBox(VectorDouble* position)
{
	if (VectorDouble((pos).getX() - (*position).getX(), 0, (pos).getZ() - (*position).getZ()).getLength() <= radius + 0.2 &&
		(*position).getY() >= (pos).getY() && (*position).getY() <= (*position).getY() + radius + 0.2)
		return pos.getY() + radius;
	return -50000;
}