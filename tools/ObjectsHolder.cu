#include "ObjectsHolder.cuh"

__host__ __device__ ObjectsHolder::ObjectsHolder() // constructeur - définition
{
	
}

ObjectsHolder::~ObjectsHolder() // Destructeur - définition
{
#ifdef __CUDA_ARCH__ //Determine si la fonction s'execute au niveau GPU
	
#else
	if (hDrawables != (int*)0xcdcdcdcdcdcdcdcd)delete hDrawables;
	if (hLightnings)delete hLightnings;
	if (hAmbientLight)delete hAmbientLight;
	if (hSkyBoxColor)delete hSkyBoxColor;
#endif
}

__host__ __device__ ObjectsHolder & ObjectsHolder::operator=(const ObjectsHolder &)
{
	return *this;
}

ObjectsHolder::ObjectsHolder(const ObjectsHolder & obj)
{
	*this = obj;
}

__device__ void ObjectsHolder::addDDrawable(int* args)
{
	switch (args[0])
	{
	case 0:
	{
		Drawable* obj = new VoidDrawable();
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	case 1:
	{
		float x1 = *(float*)&(args[1]);
		float x2 = *(float*)&(args[2]);
		float x3 = *(float*)&(args[3]);
		float rad = *(float*)&(args[4]);
		Drawable* obj = new SphereDrawable(VectorDouble(x1, x2, x3), rad, Color6Component(args[5], args[6], args[7]));
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	case 2:
	{
		float a1 = *(float*)&(args[1]);
		float a2 = *(float*)&(args[2]);
		float a3 = *(float*)&(args[3]);
		float b1 = *(float*)&(args[4]);
		float b2 = *(float*)&(args[5]);
		float b3 = *(float*)&(args[6]);
		float c1 = *(float*)&(args[7]);
		float c2 = *(float*)&(args[8]);
		float c3 = *(float*)&(args[9]);
		Drawable* obj = new TriangleDrawable(VectorDouble(a1, a2, a3), VectorDouble(b1, b2, b3), VectorDouble(c1, c2, c3), Color6Component(args[10], args[11], args[12]));
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	case 3:
	{
		float a1 = *(float*)&(args[1]);
		float a2 = *(float*)&(args[2]);
		float a3 = *(float*)&(args[3]);
		float b1 = *(float*)&(args[4]);
		float b2 = *(float*)&(args[5]);
		float b3 = *(float*)&(args[6]);
		float c1 = *(float*)&(args[7]);
		float c2 = *(float*)&(args[8]);
		float c3 = *(float*)&(args[9]);
		Drawable* obj = new TriangleMirrorDrawable(VectorDouble(a1, a2, a3), VectorDouble(b1, b2, b3), VectorDouble(c1, c2, c3), Color6Component(args[10], args[11], args[12]));
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	default:
	{
		Drawable* obj = new VoidDrawable();
		int* tempsize = new int;
		*tempsize = dDsize;
		dInts->add(tempsize);
		dDrawables->add(obj);
		dDsize = dDrawables->getSize();
		break;
	}
	}

	return;
	
}

__device__ void ObjectsHolder::addDLightning(DirectLightning* arg)
{
	dLightnings->add(arg);
	dLsize = dLightnings->getSize();
}

__device__ void ObjectsHolder::modifyDDrawable(int index, int* arg)
{
	switch (arg[0])
	{
	case 0:
	{
		VoidDrawable* obj = new VoidDrawable();
		delete dDrawables->at(index);
		dDrawables->set(index, obj);
		break;
	}
	case 1:
	{
		float x1 = *(float*)&(arg[1]);
		float x2 = *(float*)&(arg[2]);
		float x3 = *(float*)&(arg[3]);
		float rad = *(float*)&(arg[4]);
		SphereDrawable* obj = new SphereDrawable(VectorDouble(x1, x2, x3), rad, Color6Component(arg[5], arg[6], arg[7]));
		delete dDrawables->at(index);
		dDrawables->set(index, obj);
		break;
	}
	default:
	{
		VoidDrawable* obj = new VoidDrawable();
		delete dDrawables->at(index);
		dDrawables->set(index, obj);
		break;
	}
	}

	return;
}

__device__ void ObjectsHolder::modifyDLightning(int index, DirectLightning* arg)
{
	if (index <= dLsize) {
		delete(dLightnings->at(index));
		dLightnings->set(index, arg);
	}
	return;
}


__device__ Drawable* ObjectsHolder::get_d(unsigned int index)
{
	return dDrawables->operator[](index);
}

__device__ DirectLightning * ObjectsHolder::get_l(unsigned int index)
{
	return dLightnings->at(index);
}

__host__ void ObjectsHolder::initFromHostSide() {

	hDsize = 0;
	hLsize = 0;
	hAmbientLight = (int*)malloc(3*sizeof(int));
	hSkyBoxColor = (int*)malloc(3 * sizeof(int));
}

__device__ void ObjectsHolder::endFromDevice()
{
	for (int n = 0; n < dDsize; n++) {
		Drawable* temp = dDrawables->at(n);
		delete temp;
	}
	for (int n = 0; n < dLsize; n++) {
		DirectLightning* temp = dLightnings->at(n);
		delete temp;
	}

	delete dDrawables;
	delete dLightnings;
	delete dInts;
	if (dAmbientlight) delete dAmbientlight;
}

__device__ void ObjectsHolder::initFromDeviceSide() {
	dDsize = 0;
	dLsize = 0;
	dActualSizeX = 2;
	dActualSizeY = 2;
	dFactor = VectorDouble();
	dSkyBoxColor = Color6Component();

	dDrawables = new DeviceList<Drawable*>();
	dLightnings = new DeviceList<DirectLightning*>();
	dInts = new DeviceList<int*>();

	dLightnings->add(new PointDirectLightning(VectorDouble(0,20,0),Color6Component(8000,8000,8000),100));
	dLsize = dLightnings->getSize();

	dAmbientlight = new AmbientLightning();
	dAmbientlight->setLight(Color6Component());
}

__global__ void transfertNDrawable(ObjectsHolder* gpuObj, int* arg)
{
	gpuObj->addDDrawable(arg);
}

__device__ Color6Component ObjectsHolder::getLightValueAt(VectorDouble* pos)
{
	Color6Component result = dAmbientlight->getLight();
	for (int i = 0; i < dLsize; i++) {
		VectorDouble dir = dLightnings->at(i)->getPos().sub(*pos).unitVector();
		Ray ray = Ray(*pos, dir);
		bool hit = false;
		for (int j = 0; j < dDsize; j++) {
			hit = !dDrawables->at(j)->hit(ray, 0.001, pos->sub(ray.getOrigin()).getLength() - 0.001).isEmpty;
			if (hit) break;
		}
		if (!hit) {
			result = result.add(dLightnings->at(i)->getLightByDistance(pos->sub(ray.getOrigin()).getLength()));
		}
	}
	return result;
}

__device__ Color6Component ObjectsHolder::hit(Ray * r, double tmin, double tmax, int rmax)
{
	int reflect = 0;
	bool hit = false;
	bool recalculate;
	double closest = tmax;
	Ray tempRay, actualray;
	actualray = *r;
	tempRay = Ray();
	VectorDouble factor = VectorDouble(1.0, 1.0, 1.0);

	HitRecord localHit;
	Color6Component localColor;
	Color6Component localLight;

	do {
		recalculate = false;
		for (int i = 0; i < dDsize; i++) {
			HitRecord temp = HitRecord();
			temp = dDrawables->at(i)->hit(actualray, tmin, closest);
			Drawable* rtghlfguh = dDrawables->at(i);
			if (!temp.isEmpty) {
				if (dDrawables->at(i)->doReflect()) {
					recalculate = true;
					tempRay = Ray(temp.point, temp.normal);
				}
				else {
					hit = true;
				}
				closest = temp.t;
				localHit = temp;
				Color6Component tempC = dDrawables->at(i)->getColor(&temp);
				localColor = Color6Component(tempC.rComponent*factor.getX(), tempC.gComponent*factor.getY(), tempC.bComponent*factor.getZ());
				localLight = getLightValueAt(&temp.point);
			}
		}
		if (recalculate) {
			factor.setX(localColor.rComponent / 32767);
			factor.setY(localColor.gComponent / 32767);
			factor.setZ(localColor.bComponent / 32767);
			actualray = tempRay;
			closest = tmax - closest;
			reflect++;
			hit = false;
		}
	} while (recalculate && reflect < rmax);
	if (!hit) {
		double t = 0.5 * (actualray.getDirection().unitVector().getY() + 1.0);
		localColor = Color6Component(((1.0 - t) + t * dSkyBoxColor.rComponent), ((1.0 - t) + t * dSkyBoxColor.gComponent), ((1.0 - t) + t * dSkyBoxColor.bComponent));
	}
	else {
		localColor = localColor.add(localLight);
	}
	return localColor;
}

__global__ void transfertNLightnings(ObjectsHolder* gpuObj, int* arg)
{
	switch (arg[0])
	{
	case 0:
	{
		VoidDirectLightning* obj = new VoidDirectLightning();
		gpuObj->addDLightning(obj);
		break;
	}
	case 1:
	{
		float x1 = *(float*)&(arg[1]);
		float x2 = *(float*)&(arg[2]);
		float x3 = *(float*)&(arg[3]);
		float att = *(float*)&(arg[7]);
		PointDirectLightning* obj = new PointDirectLightning(VectorDouble(x1, x2, x3), Color6Component(arg[4], arg[5], arg[6]), att);
		gpuObj->addDLightning(obj);
		break;
	}
	default:
	{
		VoidDirectLightning* obj = new VoidDirectLightning();
		gpuObj->addDLightning(obj);
		break;
	}
	}

	return;
}

__global__ void transfertEDrawable(ObjectsHolder* gpuObj, int* arg, int index)
{
	gpuObj->modifyDDrawable(index, arg);
	return;
}

__global__ void transfertELightnings(ObjectsHolder* gpuObj, int* arg, int index)
{
	switch (arg[0])
	{
	case 0:
	{
		VoidDirectLightning* obj = new VoidDirectLightning();
		gpuObj->modifyDLightning(index, obj);
		break;
	}
	case 1:
	{
		float x1 = *(float*)&(arg[1]);
		float x2 = *(float*)&(arg[2]);
		float x3 = *(float*)&(arg[3]);
		float att = *(float*)&(arg[7]);
		PointDirectLightning* obj = new PointDirectLightning(VectorDouble(x1, x2, x3), Color6Component(arg[4], arg[5], arg[6]), att);
		gpuObj->modifyDLightning(index, obj);
		break;
	}
	default:
	{
		VoidDirectLightning* obj = new VoidDirectLightning();
		gpuObj->modifyDLightning(index, obj);
		break;
	}
	}

	return;
}

__global__ void transfertSkyBoxColor(ObjectsHolder* gpuObj, int* arg)
{
	Color6Component obj = Color6Component(arg[0], arg[1], arg[2]);
	gpuObj->dSkyBoxColor = obj;
	return;
}

__global__ void transfertAmbientLight(ObjectsHolder* gpuObj, int* arg)
{
	AmbientLightning* obj = new AmbientLightning(Color6Component(arg[0], arg[1], arg[2]));
	if (gpuObj->dAmbientlight) {
		delete gpuObj->dAmbientlight;
	}
	gpuObj->dAmbientlight = obj;
	return;
}

__global__ void transfertFactor(ObjectsHolder* gpuObj, int* arg)
{
	float x = *(float*)&(arg[0]);
	float y = *(float*)&(arg[1]);
	float z = *(float*)&(arg[2]);
	VectorDouble obj = VectorDouble(x, y, z);
	gpuObj->dFactor = obj;
	return;
}

__host__ int ObjectsHolder::addDrawable(ObjectsHolder* gpuObj, int * arg)
{
	if (hDsize == 0) {
		hDrawables = (int*)malloc(objectSizeA * sizeof(int));
	}
	else {
		hDrawables = (int*)realloc(hDrawables, (hDsize+1) * objectSizeA * sizeof(int));
	}
	hDsize++;
	for (int j = 0; j < objectSizeA; j++) {
		hDrawables[(hDsize - 1)*objectSizeA + j] = arg[j];
	}
	
	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, objectSizeA * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, objectSizeA * sizeof(int), cudaMemcpyHostToDevice));
	transfertNDrawable <<<1, 1>>> (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return hDsize-1;
}

__host__ int ObjectsHolder::addLightning(ObjectsHolder* gpuObj, int * arg)
{
	if (hLsize == 0) {
		hLightnings = (int*)malloc(objectSizeB * sizeof(int));
	}
	else {
		hLightnings = (int*)realloc(hLightnings, (hLsize + 1) * objectSizeB * sizeof(int));
	}
	hLsize++;
	for (int j = 0; j < objectSizeB; j++) {
		hLightnings[(hLsize-1)*objectSizeB + j] = arg[j];
	}

	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, objectSizeB * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, objectSizeB * sizeof(int), cudaMemcpyHostToDevice));
	transfertNLightnings <<<1, 1 >>> (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return hLsize-1;
}

__host__ int ObjectsHolder::modifyDrawable(ObjectsHolder* gpuObj, int index, int * arg)
{
	if (index > hDsize) {
		return -1;
	}
	for (int j = 0; j < objectSizeA; j++) {
		hDrawables[index*objectSizeA + j] = arg[j];
	}
	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, objectSizeA * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, objectSizeA * sizeof(int), cudaMemcpyHostToDevice));
	transfertEDrawable <<<1, 1 >>> (gpuObj, gpuArgs, index);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return index;
}

__host__ int ObjectsHolder::modifyLightning(ObjectsHolder* gpuObj, int index, int * arg)
{
	if (index > hLsize) {
		return -1;
	}
	for (int j = 0; j < objectSizeB; j++) {
		hLightnings[index*objectSizeB + j] = arg[j];
	}
	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, objectSizeB * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, objectSizeB * sizeof(int), cudaMemcpyHostToDevice));
	transfertELightnings << <1, 1 >> > (gpuObj, gpuArgs, index);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return index;
}

__host__ void ObjectsHolder::setSkyBoxColor(ObjectsHolder * gpuObj, int* arg)
{
	if (hSkyBoxColor) {
		delete hSkyBoxColor;
	}
	hSkyBoxColor = (int*)malloc(3 * sizeof(int));
	hSkyBoxColor[0] = arg[0];
	hSkyBoxColor[1] = arg[1];
	hSkyBoxColor[2] = arg[2];

	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, 3 * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, 3 * sizeof(int), cudaMemcpyHostToDevice));
	transfertSkyBoxColor <<<1, 1 >>> (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return;
}

__host__ void ObjectsHolder::setAmbientLight(ObjectsHolder * gpuObj, int* arg)
{
	if (hAmbientLight) {
		delete hAmbientLight;
	}
	hAmbientLight = (int*)malloc(3 * sizeof(int));
	hAmbientLight[0] = arg[0];
	hAmbientLight[1] = arg[1];
	hAmbientLight[2] = arg[2];

	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, 3 * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, arg, 3 * sizeof(int), cudaMemcpyHostToDevice));
	transfertAmbientLight <<<1, 1 >>> (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	return;
}

__host__ void ObjectsHolder::setFactor(ObjectsHolder * gpuObj, VectorDouble arg)
{
	int* tempArg = (int*)malloc(3 * sizeof(int));
	float d0 = arg.getX();
	tempArg[0] = *(int*)&(d0);
	float d1 = arg.getX();
	tempArg[1] = *(int*)&(d1);
	float d2 = arg.getX();
	tempArg[2] = *(int*)&(d2);

	int* gpuArgs;
	gpuErrchk(cudaMalloc((void**)&gpuArgs, 3 * sizeof(int)));
	gpuErrchk(cudaMemcpy(gpuArgs, tempArg, 3 * sizeof(int), cudaMemcpyHostToDevice));
	transfertFactor <<<1, 1 >>> (gpuObj, gpuArgs);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(gpuArgs);
	free(tempArg);
	return;
}