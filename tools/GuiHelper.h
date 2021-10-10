#pragma once

#include <cstdio>
#include <cstdlib>

static bool** chars;
static int charSX, charSY;

void renderGUI(int*,int,bool);
int checkCom(unsigned char*,int);
int charToInt(unsigned char*,int*);
char* intToChar(int);
void drawString(int*,const char*,int,int,int);
void drawTex(int*,unsigned char*,int,int);
int initChars();
int loadTexture(unsigned char**, const char*);
void endChars();