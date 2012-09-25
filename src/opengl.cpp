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
#include "kmeansw.h"

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
    const auto &pos = light.get_position();
    const auto &diffuse = light.get_diffuse();
    const auto &specular = light.get_specular();
    glTranslatef(pos.at(0), pos.at(1), pos.at(2));
    glColor3f(diffuse.at(0)+specular.at(0), diffuse.at(1)+specular.at(1), diffuse.at(2)+specular.at(2));
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

std::tuple<cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<cv::Vec3f>, cv::Mat_<float> > renderSceneIntoFBO() {
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

    return std::make_tuple(image, normals, position, diffuse, specular, depth);
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

template<typename T>
double sum(const T& v) {
  double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
  return sum;
}

bool test_sum() {
  bool ret_val = true;
  float numbers[] = {1,2,3,4};
  double numbers_sum = sum(numbers);
  ret_val &= numbers_sum == 10.0f;
  std::vector<float> vec;
  vec.push_back(10);
  vec.push_back(20);
  vec.push_back(30);
  vec.push_back(40);
  vec.push_back(50);
  numbers_sum = sum(vec);
  ret_val &= numbers_sum == 150;
  
  vec = std::vector<float>();
  vec.push_back(0.1);
  vec.push_back(0.3);
  vec.push_back(0.5);
  vec.push_back(0.7);
  vec.push_back(-0.1);
  numbers_sum = sum(vec);
  ret_val &= numbers_sum == 1.5;
  return ret_val;
}

template<int dim, typename T>
Lights<T> reduce_lights(const Lights<T>& lights, const unsigned int k) {
  cv::Mat_<cv::Vec<T, dim>> positions(lights.lights.size(), 1);
  std::vector<double> weight(positions.rows);
  for (int i = 0; i < positions.rows; i++) {
    const Light<T>& light = lights.lights.at(i);
    cv::Vec<T, dim> pos;
    for (unsigned int j = 0; j < dim; j++)
      pos[j] = light.get_position().at(j);
    positions(i) = pos;
//    weight.at(i) = sum(light.get_diffuse()) + sum(light.get_specular());
    // RGB for diffuse and specular -> 6 values from 0 to 1
    // let sum range from 0 to 2
    weight.at(i) = std::pow(20, 2.0/6*(sum(light.get_diffuse()) + sum(light.get_specular())));

//    std::cout << "light position: " << pos << ", weight: " << weight.at(i) << std::endl;
  }
  cv::Mat labels;
  cv::TermCriteria termcrit(cv::TermCriteria::EPS, 1000, 0.01);
  cv::Mat centers;
  cv::kmeansw(positions, k, labels, termcrit, 1, cv::KMEANS_RANDOM_CENTERS, centers, weight);

  cv::Mat_<cv::Vec<T, dim>> centers_templ(k, 1);
  for (int i = 0; i < centers.rows; i++) {
    centers_templ(i) = centers.at<cv::Vec<T, dim>>(i);
  }
  
  return Lights<T>(centers_templ);
}

void calc_lights() {
  assert(test_sum());

//  testkmeansall();

  const auto start_time = std::chrono::high_resolution_clock::now();
  
  cv::Mat_<cv::Vec3f> image;
  cv::Mat_<cv::Vec3f> normals;
  cv::Mat_<cv::Vec3f> position;
  cv::Mat_<cv::Vec3f> diffuse;
  cv::Mat_<cv::Vec3f> specular;
  
  std::tie(image, normals, position, diffuse, specular, std::ignore) = create_test_image();
  const auto test_creation_time = std::chrono::high_resolution_clock::now();
  std::cout << "test created" << std::endl;
  
  // do not need to be flipped
  GLfloat model_view_matrix_stack[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, model_view_matrix_stack);
  cv::Mat_<GLfloat> model_view_matrix(4, 4, model_view_matrix_stack);

  const unsigned int huge_num_lights = 20;
  const unsigned int small_num_lights = 10;
  float x, y, z;
  std::tie(x, y, z) = _ball.getViewDirection();
  Lights<float> a_lot_of_lights("bla", 10, huge_num_lights, plane_acceptor_tuple(cv::Vec3f(-x, -y, -z), cv::Vec3f(0, 0, 0)));
  const auto time_after_huge_lights_creation = std::chrono::high_resolution_clock::now();
  std::cout << "a lot of lights created" << std::endl;

  optimize_lights<ls>(image, normals, position, diffuse, specular, model_view_matrix.t(), clear_color, lights);
  optimize_lights<multi_dim_fit>(image, normals, position, diffuse, specular, model_view_matrix.t(), clear_color, a_lot_of_lights);
  optimize_lights<nnls_struct>(image, normals, position, diffuse, specular, model_view_matrix.t(), clear_color, a_lot_of_lights);
  const auto time_after_huge_lights_run = std::chrono::high_resolution_clock::now();
  std::cout << "a lot of lights optimized" << std::endl;

  if (!_args.single_pass) {
    lights = reduce_lights<4>(a_lot_of_lights, small_num_lights);
    std::cout << "a lot of lights reduced" << std::endl;

//    optimize_lights<ls>(image, normals, position, diffuse, specular, model_view_matrix.t(), clear_color, lights);
//    optimize_lights<multi_dim_fit>(image, normals, position, diffuse, specular, model_view_matrix.t(), clear_color, lights);
    optimize_lights<nnls_struct>(image, normals, position, diffuse, specular, model_view_matrix.t(), clear_color, lights);

    
    std::cout << "small number of lights reduced" << std::endl;
  } else {
    lights = a_lot_of_lights;
  }
  
  const auto finish_time = std::chrono::high_resolution_clock::now();

  std::cout << "complete run: " << finish_time - start_time << std::endl;
  std::cout << "  test creation: " << test_creation_time - start_time << std::endl;
  std::cout << "  huge light number creation: " << time_after_huge_lights_creation - test_creation_time << std::endl;
  std::cout << "  light estimation huge light number: " << time_after_huge_lights_run - time_after_huge_lights_creation << std::endl;
  std::cout << "  light estimation smaller light number: " << finish_time - time_after_huge_lights_run << std::endl;
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
    calc_lights();
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
  lights = Lights<float>(light_properties, sizeof(light_properties)/sizeof(light_properties[0])/NUM_PROPERTIES);
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
