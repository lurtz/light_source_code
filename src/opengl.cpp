#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <iterator>
#include <tuple>
#include "opengl.h"
#include "Trackball.h"

#ifdef OPENCV_OLD_INCLUDES
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/highgui/highgui.hpp>
#endif

#include "solver.h"

MeshObj * _meshobj = nullptr;
cv::Mat const * _original_image = nullptr;
Trackball _ball;

GLclampf clear_color = 0.3;
GLfloat _zNear, _zFar;
GLfloat _fov;

// FBO stuff //
GLuint fboTexture[3];
GLuint fboDepthTexture;
GLuint fbo;

std::vector<float> ambient;
std::vector<Light<float>::properties> lights;

bool image_displayed = false;

#define BUFFER_OFFSET(i) ((char *)nullptr + (i))

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

    if (false)
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

template<class RandomAccessIterator>
void flipImage(RandomAccessIterator first_row, RandomAccessIterator past_last_row, const unsigned int width) {
  for (; first_row < past_last_row; first_row+=width, past_last_row-=width) {
    std::swap_ranges(first_row, first_row + width, past_last_row - width);
  }
}

// this nice friend works with std::vector<> as well as for cv::Mat_<>
template<class T>
void flipImage(T& image, const unsigned int width) {
  flipImage(std::begin(image), std::end(image), width);
}

std::tuple<cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f> > renderSceneIntoFBO() {
    image_displayed = true;
    // render scene into first color attachment of FBO -> use as filter texture later on //
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    static const GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, buffers);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    renderScene();

    // TODO problem: value of FBO textures ranges only from 0 to 1
    // read data from frame buffer
    // create opencv images
    cv::Mat_<cv::Vec3f> image(windowHeight, windowWidth);
    cv::Mat_<cv::Vec3f> normals(windowHeight, windowWidth);
    cv::Mat_<cv::Vec3f> position(windowHeight, windowWidth);

    // read each texture into an array
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_BGR, GL_FLOAT, image.data);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_BGR, GL_FLOAT, normals.data);
    glReadBuffer(GL_COLOR_ATTACHMENT2);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_BGR, GL_FLOAT, position.data);
    glReadBuffer(GL_DEPTH_ATTACHMENT);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // opencv images are upside down
    flipImage(image, windowWidth);
    flipImage(normals, windowWidth);
    flipImage(position, windowWidth);

    // convert normals from [0,1] to [-1,1]
    normals = (normals*2) -1;

    return std::make_tuple(image, normals, position);
}

std::tuple<cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f> > create_test_image() {
  auto tmp_lights = create_lights_from_array(light_properties, sizeof(light_properties)/sizeof(light_properties[0])/NUM_PROPERTIES);
  _meshobj->setLight(ambient, tmp_lights);

  auto tuple = renderSceneIntoFBO();

  _meshobj->setLight(ambient, lights);
  return tuple;
}

void calc_lights() {
  if (image_displayed)
    return;

  cv::Mat_<cv::Vec3f> image;
  cv::Mat_<cv::Vec3f> normals;
  cv::Mat_<cv::Vec3f> position;
  std::tie(image, normals, position) = create_test_image();

#if false
    cv::imshow("fbo texture", image);
    cv::imshow("normals2", normals);
    cv::imshow("position", position);
    cv::waitKey(100);
#else

  GLfloat model_view_matrix[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, model_view_matrix);
  cv::Mat_<GLfloat> model_view_matrix_cv(4, 4, model_view_matrix, 0);

  optimize_lights<float>(image, normals, position, model_view_matrix_cv, clear_color, ambient, lights);
#endif
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
  calc_lights();
  renderScene();
  
  // swap render and screen buffer //
  glutSwapBuffers();
}

void run(const cv::Mat& original_image, MeshObj * const meshobj) {
  _original_image = &original_image;
  _meshobj = meshobj;
  _meshobj->setLight(ambient, lights);
  glutMainLoop();
//  glutMainLoopEvent();
}

void keyboardEvent(unsigned char key, int x, int y) {
  switch (key) {
    case 'x':
    case 27 : {
      glutLeaveMainLoop();
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
  glClearColor(clear_color, clear_color, clear_color, clear_color);
  glEnable(GL_DEPTH_TEST);

  // set projectionmatrix
  glMatrixMode(GL_PROJECTION);
  gluPerspective(_fov, static_cast<GLdouble>(windowWidth)/windowHeight, _zNear, _zFar);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void initFBO() {
  // init color textures //
  glGenTextures(3, fboTexture);
  for (unsigned int i = 0; i < 3; ++i) {
    glBindTexture(GL_TEXTURE_2D, fboTexture[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
  }
  // init depth texture //
  glGenTextures(1, &fboDepthTexture);
  glBindTexture(GL_TEXTURE_2D, fboDepthTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

  // generate FBO and depthBuffer //
  glGenFramebuffers(1, &fbo);

  // attach textures to FBO //
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTexture[0], 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, fboTexture[1], 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, fboTexture[2], 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fboDepthTexture, 0);

  // unbind FBO until it's needed //
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void initLights() {
  ambient = create_ambient_color<float>();
//  lights = create_lights_from_array(light_properties, sizeof(light_properties)/sizeof(light_properties[0])/NUM_PROPERTIES);
  lights = create_light_sphere<float>(10, 20);
  if (_meshobj != nullptr)
	  _meshobj->setLight(ambient, lights);
}

void setupOpenGL(int * argc, char ** argv, const unsigned int width, const unsigned int height) {
    /* Initialize GLUT */
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(windowWidth = width, windowHeight = height);
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
    initLights();
}
