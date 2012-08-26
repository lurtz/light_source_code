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
#include <memory>
#include <chrono>
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

Lights<float> lights;

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

void visualize_lights() {
  for (const auto& light : lights.lights) {
    glPushMatrix();
    std::vector<float> pos = light.get_position();
    glTranslatef(pos.at(0), pos.at(1), pos.at(2));
    glutSolidSphere(.10, 4, 4);
    glPopMatrix();
  }
}

void renderScene() {
    _meshobj->render();
    if (image_displayed)
      visualize_lights();

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

std::tuple<cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<float> > renderSceneIntoFBO() {
    // render scene into first color attachment of FBO -> use as filter texture later on //
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    static const GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, buffers);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    renderScene();

    // read data from frame buffer
    // create opencv images
    cv::Mat_<cv::Vec3f> image(windowHeight, windowWidth);
    cv::Mat_<cv::Vec3f> normals(windowHeight, windowWidth);
    cv::Mat_<cv::Vec3f> position(windowHeight, windowWidth);
    cv::Mat_<float> depth(windowHeight, windowWidth);

    // read each texture into an array
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_FLOAT, image.data);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_FLOAT, normals.data);
    glReadBuffer(GL_COLOR_ATTACHMENT2);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_FLOAT, position.data);
    glReadBuffer(GL_DEPTH_ATTACHMENT);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // opencv images are upside down
    if (true) {
      flipImage(image, windowWidth);
      flipImage(normals, windowWidth);
      flipImage(position, windowWidth);
      flipImage(depth, windowWidth);
    }

    return std::make_tuple(image, normals, position, depth);
}

decltype(renderSceneIntoFBO()) create_test_image() {
  auto tmp_lights = Lights<float>(light_properties, sizeof(light_properties)/sizeof(light_properties[0])/NUM_PROPERTIES);
  _meshobj->setLight(tmp_lights);

  auto tuple = renderSceneIntoFBO();

  _meshobj->setLight(lights);
  return tuple;
}

template<class Rep, class Period>
std::ostream& operator<<(std::ostream& out, const std::chrono::duration<Rep, Period>& tp) {
  out << std::chrono::duration_cast<std::chrono::seconds>(tp).count() << "s";
  return out;
}

void calc_lights() {
  if (image_displayed)
    return;

  const auto start_time = std::chrono::steady_clock::now();
  
  cv::Mat_<cv::Vec3f> image;
  cv::Mat_<cv::Vec3f> normals;
  cv::Mat_<cv::Vec3f> position;
  std::tie(image, normals, position, std::ignore) = create_test_image();
  const auto test_creation_time = std::chrono::steady_clock::now();
  
  // do not need to be flipped
  GLfloat model_view_matrix_stack[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, model_view_matrix_stack);
  cv::Mat_<GLfloat> model_view_matrix(4, 4, model_view_matrix_stack);
  
//  optimize_lights<float>(image, normals, position, model_view_matrix.t(), clear_color, lights);
  
  optimize_lights_multi_dim_fit<float>(image, normals, position, model_view_matrix.t(), clear_color, lights);
  const auto finish_time = std::chrono::steady_clock::now();

  std::cout << "complete run: " << finish_time - start_time << std::endl;
  std::cout << "  test creation: " << test_creation_time - start_time << std::endl;
  std::cout << "  light estimation: " << finish_time - test_creation_time << std::endl;
  
  image_displayed = true;
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
  _meshobj->setLight(lights);
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
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
//  lights = Lights(light_properties, sizeof(light_properties)/sizeof(light_properties[0])/NUM_PROPERTIES);
//  lights = Lights<float>(10, 140);
  float x, y, z;
  std::tie(x, y, z) = _ball.getViewDirection();
//  lights = Lights<float>(10, 30);
//  lights = Lights<float>(10, 30, plane_acceptor(cv::Vec3f(-x, -y, -z), cv::Vec3f(0, 0, 0)));
  lights = Lights<float>("bla", 10, 140, plane_acceptor_tuple(cv::Vec3f(-x, -y, -z), cv::Vec3f(0, 0, 0)));
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
