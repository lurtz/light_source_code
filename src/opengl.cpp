#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <limits>
#include "opengl.h"

MeshObj const * _meshobj;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

// Define some globals
int windowWidth, windowHeight;

float camX, camY, camZ = 4.0f; //X, Y, and Z
float lastx, lasty, xrot, yrot, xrotrad, yrotrad; //Last pos and rotation
float speed = 0.1f; //Movement speed

void mouseMovement(int x, int y) {
    int cx = windowWidth/2;
    int cy = windowHeight/2;

    if(x == cx && y == cy){ //The if cursor is in the middle
        return;
    }

    int diffx=x-cx; //check the difference between the current x and the last x position
    int diffy=y-cy; //check the difference between the current y and the last y position
    xrot += (float)diffy/2; //set the xrot to xrot with the addition of the difference in the y position
    yrot += (float)diffx/2;// set the xrot to yrot with the addition of the difference in the x position
    glutWarpPointer(cx, cy); //Bring the cursor to the middle
}

void idle() {
    glutPostRedisplay();
}

void reshape(int width, int height) {
    windowWidth = width;
    windowHeight = height;
    glMatrixMode(GL_PROJECTION);
    glViewport(0, 0, width, height);
    gluPerspective(45.0f, (GLfloat)width/(GLfloat)height, 0.5f, 10000.0f);
}

void renderScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // TODO some camera things with gluLookAt()
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glutSwapBuffers();
}

void run() {
    glutMainLoop();
}

void keyboard(unsigned char key, int x, int y) {
    switch(key) {
        //WASD movement
        case 'w':
            camZ -= 0.1f;
        break;
        case 's':
            camZ += 0.1f;
        break;
        case 'a':
            camX -= 0.1f;
            break;
        case 'd':
            camX += 0.1f;
        break;
        case 27:
            //TODO some clean up
            glutLeaveMainLoop();
        break;
        default:
            std::cout << static_cast<char>(key) << std::endl;
        }
}

void setupOpenGL(int * argc, char ** argv, MeshObj const * const meshobj) {
    /* Initialize GLUT */
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(windowWidth = 800, windowHeight = 600);
    glutCreateWindow("light sources");
//    glutFullScreen();
    glutDisplayFunc(renderScene);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(mouseMovement);

    glewInit();
    glEnable(GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);

    _meshobj = meshobj;
}
