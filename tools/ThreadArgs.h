#pragma once
#include "../windowsUtil/WindowsHolder.h"
#include "ObjectsHolder.cuh"

class ThreadArgs
{
public:
	ThreadArgs();
	ThreadArgs(WindowsHolder* wh, ObjectsHolder* oh);

	WindowsHolder* getWHolder();
	ObjectsHolder* getOHolder();

private:

	WindowsHolder* wHolder;
	ObjectsHolder* oHolder;
};