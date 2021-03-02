#include <windows.h>
#include "WindowsHolder.h"
#include "VectorDouble.cuh"
#include <cmath>
#include <corecrt_math_defines.h>

extern void setResolution(int x, int y);
extern void initRayTracer();
extern int* renderImage();
extern void setCameraArgs(VectorDouble arg1, VectorDouble arg2, VectorDouble arg3, double arg4);
extern void endRayTracer();
extern int addDrawableObj(int* arg);
extern int addLightningObj(int* arg);
extern void setSkyBoxColorRGB(int* arg);
extern void setInitialised();

static WindowsHolder* windowC;

static VectorDouble targetBase = VectorDouble(1, -0.5, 0);
static VectorDouble camBase = VectorDouble(2, 0, 0);

static int wWidth = 100;
static int wHeight = 100;

static constexpr double speedBase = 0.07;
static constexpr double speedCrouch = 0.025;

static double rotX = 0;
static double rotY = 0;

static bool refreshScreenSize() {
	int maxX = windowC->getDrawSizeX();
	int maxY = windowC->getDrawSizeY();

	if (wWidth != maxX || wHeight != maxY) {
		if (maxX != 0 && maxY != 0) {
			wWidth = maxX;
			wHeight = maxY;
			setResolution(wWidth, wHeight);
			return true;
		}
	}
	return false;

}

static double toRadians(double angle) {
	return angle/180*M_PI;
}

static double cut(double value, double min, double max) {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

static void refreshRotation() {
	setCameraArgs(camBase, VectorDouble(cos(toRadians(rotX))*cos(toRadians(rotY)), sin(toRadians(rotY)), sin(toRadians(rotX))*cos(toRadians(rotY))), VectorDouble(0, 1, 0), 90);
}

static void refreshScreenValue() {
	int* screen = renderImage();
	windowC->setOutputScreen(screen);
}

static VectorDouble getPlayerDir() {
	VectorDouble result = VectorDouble(0, 0, 0);
	const double speed = windowC->getInputValue(VK_CONTROL)?speedCrouch:speedBase;
	if (windowC->getInputValue('Z')) {
		result.setX(1.0);
	}
	if (windowC->getInputValue('S')) {
		result.setX(result.getX() - 1.0);
	}
	if (windowC->getInputValue('D')) {
		result.setZ(1.0);
	}
	if (windowC->getInputValue('Q')) {
		result.setZ(result.getZ() - 1.0);
	}
	if (result.length_squared() > 0.001) {
		result = result.unitVector().mul(speed);
	}
	return result;
}

static void initClient() {
	initRayTracer();
	setCameraArgs(VectorDouble(2, 0, 0), VectorDouble(1, -0.5, 0), VectorDouble(0, 1, 0), 90);
	
	int* obj1 = (int*)malloc(13*sizeof(int));
	obj1[0] = 1;
	float posX = 0;
	float posY = 0;
	float posZ = 1;
	float rad = 0.7;
	obj1[1] = *(int*)&(posX);
	obj1[2] = *(int*)&(posY);
	obj1[3] = *(int*)&(posZ);
	obj1[4] = *(int*)&(rad);
	obj1[5] = 0;
	obj1[6] = 16000;
	obj1[7] = 0;
	addDrawableObj(obj1);

	posX = 0;
	posY = 0.2;
	posZ = -1;
	rad = 0.3;
	obj1[1] = *(int*)&(posX);
	obj1[2] = *(int*)&(posY);
	obj1[3] = *(int*)&(posZ);
	obj1[4] = *(int*)&(rad);
	obj1[5] = 8000;
	obj1[6] = 0;
	obj1[7] = 16000;
	addDrawableObj(obj1);
	free(obj1);

	int* obj2 = (int*)malloc(13 * sizeof(int));
	obj2[0] = 2;
	float A1, A2, A3, B1, B2, B3, C1, C2, C3;
	A1 = -1;
	A2 = -0.5;
	A3 = -1;
	B1 = -1;
	B2 = -1;
	B3 = 1;
	C1 = 1;
	C2 = -1;
	C3 = 0;
	obj2[1] = *(int*)&(A1);
	obj2[2] = *(int*)&(A2);
	obj2[3] = *(int*)&(A3);
	obj2[4] = *(int*)&(B1);
	obj2[5] = *(int*)&(B2);
	obj2[6] = *(int*)&(B3);
	obj2[7] = *(int*)&(C1);
	obj2[8] = *(int*)&(C2);
	obj2[9] = *(int*)&(C3);
	obj2[10] = -1;
	obj2[11] = 0;
	obj2[12] = 0;
	addDrawableObj(obj2);
	free(obj2);

	int* obj3 = (int*)malloc(8 * sizeof(int));
	obj3[0] = 1;
	float posX2 = 1;
	float posY2 = 10;
	float posZ2 = 5;
	float att = 99;
	obj3[1] = *(int*)&(posX);
	obj3[2] = *(int*)&(posY);
	obj3[3] = *(int*)&(posZ);
	obj3[7] = *(int*)&(rad);
	obj3[4] = 8000;
	obj3[5] = 8000;
	obj3[6] = 8000;
	addLightningObj(obj3);
	free(obj3);

	int* obj4 = (int*)malloc(3 * sizeof(int));
	obj4[0] = 22937;
	obj4[1] = 26214;
	obj4[2] = 32767;
	setSkyBoxColorRGB(obj4);
	free(obj4);
	setInitialised();
}

static DWORD WINAPI clientMain(LPVOID lpParameter) {
	windowC = ((WindowsHolder*)lpParameter);

	initClient();

	while (windowC->isCaptured() || !windowC->getInputValue(VK_CONTROL)) {
		refreshScreenSize();

		if (windowC->isCaptured()) {
			int test = windowC->getMousePosX();
			rotX = windowC->getMousePosX() / 5.0 + rotX;
			while (rotX < 0) rotX += 360;
			while (rotX >= 360) rotX -= 360;
			rotY = windowC->getMousePosY() / 5.0 + rotY;
			rotY = cut(rotY, -90, 90);
			VectorDouble playerDir = getPlayerDir();
			camBase.setX(camBase.getX() + cos(toRadians(rotX))*(-playerDir.getX()) + sin(toRadians(rotX))*playerDir.getZ());
			camBase.setZ(camBase.getZ() + sin(toRadians(rotX))*(-playerDir.getX()) - cos(toRadians(rotX))*playerDir.getZ());
			refreshRotation();
			windowC->setMousePosInput(0, 0);
		}
		

		if (windowC->getInputValue(VK_ESCAPE)) {
			windowC->setInputValue(VK_ESCAPE, false);
			windowC->setCaptured(!windowC->isCaptured());
		}

		refreshScreenValue();

		InvalidateRect(windowC->windowH, NULL, TRUE);
		UpdateWindow(windowC->windowH);
		windowC->sizing = false;

	}
	endRayTracer();
	windowC->exitWindow = true;

	return 0;
}