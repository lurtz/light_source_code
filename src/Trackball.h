#ifndef __TRACKBALL__
#define __TRACKBALL__

#include <GL/glut.h>
#include <cmath>

#define STEP_DISTANCE 0.01f

class Trackball {
  public:
    Trackball(float theta = 0, float phi = 0, float dist = 1);
    ~Trackball();
    
    enum MouseState {NO_BTN = 0, LEFT_BTN, RIGHT_BTN, MIDDLE_BTN};
    enum Motion {MOVE_FORWARD = 0, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT};
    
    void updateMousePos(int x, int y);
    void updateMouseBtn(MouseState state, int x, int y);
    void updateOffset(Motion motion);
    
    void reset(float theta = 0, float phi = 0, float dist = 1);
    void rotateView(void);
    
    void getCameraPosition(float &x, float &y, float &z);
    
  private:
    float mViewOffset[3];
    float mTheta, mPhi;
    float mLastTheta, mLastPhi;
    int mX, mY;
    MouseState mState;
};

#endif
