#include "ThreadArgs.h"

ThreadArgs::ThreadArgs()
{
	wHolder = nullptr;
	oHolder = nullptr;
}

ThreadArgs::ThreadArgs(WindowsHolder* wh, ObjectsHolder* oh)
{
	wHolder = wh;
	oHolder = oh;
}

WindowsHolder* ThreadArgs::getWHolder()
{
	return wHolder;
}

ObjectsHolder* ThreadArgs::getOHolder() {
	return oHolder;
}