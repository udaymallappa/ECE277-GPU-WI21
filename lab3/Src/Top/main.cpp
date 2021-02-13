
#include <string>
// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern bool initGLdevice(int argc, char **argv);

int main(int argc, char **argv)
{
	// initialize opengl
	initGLdevice(argc, argv);

	// start rendering mainloop
	glutMainLoop();
	return 0;
}