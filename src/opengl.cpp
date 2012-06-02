#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <limits>
#include "opengl.h"
#include "Trackball.h"
#include <cv.hpp>
#include <highgui.h>
#include "solver.h"


MeshObj * _meshobj;
cv::Mat const * _original_image;
Trackball _ball;

GLfloat _zNear, _zFar;
GLfloat _fov;

// FBO stuff //
GLuint fboTexture[2];
GLuint fboDepthTexture;
GLuint fbo;

// PBO
GLuint ioBuf;

std::vector<Light<float>::properties> lights;

bool image_displayed = false;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

// Define some globals
int windowWidth, windowHeight;

float camX, camY, camZ = 4.0f; //X, Y, and Z
float lastx, lasty, xrot, yrot, xrotrad, yrotrad; //Last pos and rotation
float speed = 0.1f; //Movement speed

void mouseEvent(int button, int state, int x, int y) {
  Trackball::MouseState mouseState;
  if (state == GLUT_DOWN) {
    switch (button) {
      case GLUT_LEFT_BUTTON : {
        mouseState = Trackball::LEFT_BTN;
        break;
      }
      case GLUT_RIGHT_BUTTON : {
        mouseState = Trackball::RIGHT_BTN;
        break;
      }
      default : break;
    }
  } else {
    mouseState = Trackball::NO_BTN;
  }
  _ball.updateMouseBtn(mouseState, x, y);

}

void mouseMoveEvent(int x, int y) {
  _ball.updateMousePos(x, y);
}

void idle() {
    glutPostRedisplay();
}

void reshape(int width, int height) {
    windowWidth = width;
    windowHeight = height;
    glutPostRedisplay();
}

void renderScene() {
    _meshobj->render();

    for (int x = -1; x < 2; x+=2)
        for (int y = -1; y < 2; y+=2)
            for (int z = -1; z < 2; z+=2) {
              glPushMatrix();
              glTranslatef(x, y, z);
              glColor3f((x+1.0)/3, (y+1.0)/3, (z+1.0)/3);
              glutSolidSphere(.10, 4, 4);
              glPopMatrix();
            }
}

unsigned int calc_index(const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height) {
  assert(x < width);
  assert(y < height);
  return x+y*width;
}

template <class T>
void flipImage(T * image, const unsigned int width, const unsigned int height) {
  T tmp[width];
  for (unsigned int y = 0; y < height/2; y++) {
    T * upper_line_start = image+calc_index(0, y, width, height);
    T * lower_line_start = image+calc_index(0, height - 1 - y, width, height);
    std::copy(upper_line_start, upper_line_start+width, tmp);
    std::copy(lower_line_start, lower_line_start+width, upper_line_start);
    std::copy(tmp, tmp+width, lower_line_start);
  }
}

void renderSceneIntoFBO() {
    if (image_displayed)
      return;
    image_displayed = true;
    // render scene into first color attachment of FBO -> use as filter texture later on //
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    renderScene();

    float * bla = new float[windowHeight*windowWidth*4];

#if false
    glReadBuffer(fboTexture[0]);
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, ioBuf);
    glBufferData(GL_PIXEL_PACK_BUFFER_ARB, windowWidth*windowHeight*sizeof(float)*4, NULL, GL_STREAM_READ);
    glReadPixels (0, 0, windowWidth, windowHeight, GL_BGRA, GL_FLOAT, BUFFER_OFFSET(0));
    float * mem = static_cast<float *>(glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_WRITE));
    assert(mem);
    std::copy(mem, mem+windowWidth*windowHeight*4, bla);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
#else
    glReadBuffer(fboTexture[0]);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_BGRA, GL_FLOAT, bla);
#endif
    flipImage(bla, 4*windowWidth, 1*windowHeight);

    cv::Mat image(windowHeight, windowWidth, CV_32FC4, bla, 0);
    cv::imshow("FBO texture", image);
    cv::waitKey(0);

    delete [] bla;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    optimize_lights<float>(*_original_image, image, lights);
}

void updateGL() {
  GLfloat aspectRatio = static_cast<GLfloat>(windowWidth) / windowHeight;
  
  // clear renderbuffer //
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  glViewport(0, 0, windowWidth, windowHeight);
  
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(_fov, aspectRatio, _zNear, _zFar);
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  
  _ball.rotateView();
  
  // render //
  renderScene();
//  renderSceneIntoFBO();
  
  // swap render and screen buffer //
  glutSwapBuffers();
}

void run(const cv::Mat& original_image, MeshObj * const meshobj) {
	_original_image = &original_image;
    _meshobj = meshobj;
    _meshobj->setLight(lights);
    glutMainLoop();
}

void keyboardEvent(unsigned char key, int x, int y) {
  switch (key) {
    case 'x':
    case 27 : {
      exit(0);
      break;
    }
    case 'w': {
      // move forward //
      _ball.updateOffset(Trackball::MOVE_FORWARD);
      break;
    }
    case 's': {
      // move backward //
      _ball.updateOffset(Trackball::MOVE_BACKWARD);
      break;
    }
    case 'a': {
      // move left //
      _ball.updateOffset(Trackball::MOVE_LEFT);
      break;
    }
    case 'd': {
      // move right //
      _ball.updateOffset(Trackball::MOVE_RIGHT);
      break;
    }
    default : {
      break;
    }
  }
  glutPostRedisplay();
}

void initGL() {
  glClearColor(0.3, 0.3, 0.3, 0.3);
  glEnable(GL_DEPTH_TEST);

  // set projectionmatrix
  glMatrixMode(GL_PROJECTION);
  gluPerspective(_fov, static_cast<GLdouble>(windowWidth)/windowHeight, _zNear, _zFar);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void initFBO() {
  // init color textures //
  glGenTextures(2, fboTexture);
  for (unsigned int i = 0; i < 2; ++i) {
    glBindTexture(GL_TEXTURE_2D, fboTexture[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
  }
  // init depth texture //
  glGenTextures(1, &fboDepthTexture);
  glBindTexture(GL_TEXTURE_2D, fboDepthTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

  // generate FBO and depthBuffer //
  glGenFramebuffers(1, &fbo);

  // attach textures to FBO //
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTexture[0], 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, fboTexture[1], 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fboDepthTexture, 0);

  // unbind FBO until it's needed //
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void initPBO() {
  glGenBuffers(1, &ioBuf);
}

void initLights() {
  lights = create_lights(light_properties, sizeof(light_properties)/sizeof(light_properties[0])/NUM_PROPERTIES);
  if (_meshobj != NULL)
	  _meshobj->setLight(lights);
}

void setupOpenGL(int * argc, char ** argv) {
    /* Initialize GLUT */
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(windowWidth = 800, windowHeight = 600);
    glutCreateWindow("light sources");
//    glutFullScreen();
    glutDisplayFunc(updateGL);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboardEvent);
    glutMouseFunc(mouseEvent);
    glutMotionFunc(mouseMoveEvent);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
      std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    }
    std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;

    _zNear = 0.1f;
    _zFar= 1000.0f;
    _fov = 45.0f;

    _ball.updateOffset(Trackball::MOVE_BACKWARD, 4);

    initGL();
    initFBO();
    initPBO();
    initLights();
}
