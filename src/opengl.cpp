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

arguments _args;

MeshObj * _meshobj = nullptr;
cv::Mat_<cv::Vec3f> _original_image;
Trackball _ball;

GLclampf clear_color = 0.3;
GLfloat _zNear, _zFar;
GLfloat _fov;

// FBO stuff //
// image normal position diffuse_texture specular_texture
GLuint fboTexture[5];
GLuint fboDepthTexture;
GLuint fbo;

Lights<float, 4> lights;

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
    const auto &pos = light.get<Properties::POSITION>();
    const auto &diffuse = light.get<Properties::DIFFUSE>();
    const auto &specular = light.get<Properties::SPECULAR>();
    glTranslatef(pos[0], pos[1], pos[2]);
    glColor3f(diffuse[0]+specular[0], diffuse[1]+specular[1], diffuse[2]+specular[2]);
    glutSolidSphere(.10, 4, 4);
    glPopMatrix();
  }
}

void renderScene() {
    _meshobj->render();
    if (image_displayed  || !_args.optimize)
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

std::tuple<cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<float>, cv::Mat_<GLfloat>> renderSceneIntoFBO() {
    // render scene into first color attachment of FBO -> use as filter texture later on //
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    static const GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4};
    glDrawBuffers(5, buffers);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    renderScene();

    // read data from frame buffer
    // create opencv images
    cv::Mat_<cv::Vec3f> image(windowHeight, windowWidth);
    cv::Mat_<cv::Vec3f> normals(windowHeight, windowWidth);
    cv::Mat_<cv::Vec3f> position(windowHeight, windowWidth);
    cv::Mat_<cv::Vec3f> diffuse(windowHeight, windowWidth);
    cv::Mat_<cv::Vec3f> specular(windowHeight, windowWidth);
    cv::Mat_<float> depth(windowHeight, windowWidth);

    // read each texture into an array
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_FLOAT, image.data);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_FLOAT, normals.data);
    glReadBuffer(GL_COLOR_ATTACHMENT2);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_FLOAT, position.data);
    glReadBuffer(GL_COLOR_ATTACHMENT3);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_FLOAT, diffuse.data);
    glReadBuffer(GL_COLOR_ATTACHMENT4);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_FLOAT, specular.data);
    glReadBuffer(GL_DEPTH_ATTACHMENT);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (_args.texture_filename.size() == 0)
      diffuse = specular = cv::Mat_<cv::Vec3f>(windowHeight, windowWidth, cv::Vec3f(1,1,1));
    
    // opencv images are upside down
    flipImage(image, windowWidth);
    flipImage(normals, windowWidth);
    flipImage(position, windowWidth);
    flipImage(depth, windowWidth);
    flipImage(diffuse, windowWidth);
    flipImage(specular, windowWidth);

    // do not need to be flipped
    GLfloat model_view_matrix_stack[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, model_view_matrix_stack);
    cv::Mat_<GLfloat> model_view_matrix;
    cv::Mat_<GLfloat>(4,4, model_view_matrix_stack).copyTo(model_view_matrix);

    return std::make_tuple(image, normals, position, diffuse, specular, depth, model_view_matrix);
}

decltype(renderSceneIntoFBO()) create_test_image() {
  std::cout << "creating test image..." << std::endl;
  auto tmp_lights = Lights<float, 4>(light_properties, sizeof(light_properties)/sizeof(light_properties[0])/NUM_PROPERTIES);
  _meshobj->setLight(tmp_lights);

  auto tuple = renderSceneIntoFBO();

  _meshobj->setLight(lights);
  return tuple;
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
  if (!image_displayed && _args.optimize) {
    const float radius = 10;
    const unsigned int huge_num_lights = 20;
    float x, y, z;
    std::tie(x, y, z) = _ball.getViewDirection();
    lights = Lights<float, 4>("bla", radius, huge_num_lights, plane_acceptor_tuple<float, 4>(cv::Vec4f(-x, -y, -z, 0), cv::Vec4f(0, 0, 0, 0)));
//    lights = calc_lights<ls, sample_point_random>(create_test_image(), _ball.getViewDirection(), _args);
//    lights = calc_lights<multi_dim_fit, sample_point_random>(create_test_image(), _ball.getViewDirection(), _args);
    lights = calc_lights<nnls_struct, sample_point_deterministic>(create_test_image(), lights, _args.single_pass);
    image_displayed = true;
  }
  renderScene();
  
  // swap render and screen buffer //
  glutSwapBuffers();
}

void run(MeshObj * const meshobj) {
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
  glGenTextures(5, fboTexture);
  for (unsigned int i = 0; i < 5; ++i) {
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
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, fboTexture[3], 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, fboTexture[4], 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fboDepthTexture, 0);

  // unbind FBO until it's needed //
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void initLights() {
  lights = Lights<float, 4>(light_properties, sizeof(light_properties)/sizeof(light_properties[0])/NUM_PROPERTIES);
}

void setupOpenGL(int * argc, char ** argv, const arguments &args) {
    _args = args;
    _original_image = cv::imread(args.image_filename);
    windowWidth = _original_image.cols;
    windowHeight = _original_image.rows;
    if (_original_image.rows == 0 || _original_image.cols == 0) {
      std::cerr << "INPUT DATA HAS 0 SIZE!" << std::endl;
      windowWidth = 640;
      windowHeight = 480;
      _original_image = cv::Mat_<cv::Vec3f>(windowHeight, windowWidth);
      std::cerr << "setting size to " << windowWidth << "x" << windowHeight << std::endl;
    }
  
    /* Initialize GLUT */
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(windowWidth, windowHeight);
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
