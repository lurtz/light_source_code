#include "Trackball.h"
#include <iostream>

Trackball::Trackball(float theta, float phi, float dist) {
  reset(theta, phi, dist);
}

Trackball::~Trackball() {}
    
void Trackball::updateMousePos(int x, int y) {
  switch (mState) {
    case LEFT_BTN : {
      // left button pressed -> compute position difference to click-point and compute new angles //
      mTheta = mLastTheta + (float)(mX - x) / 256;
      mPhi = mLastPhi - (float)(mY - y) / 256;
      break;
    }
    case RIGHT_BTN : {
      // not used yet //
      break;
    }
    default : break;
  }
}

void Trackball::updateMouseBtn(MouseState state, int x, int y) {
  switch (state) {
    case NO_BTN : {
      // button release -> save current angles for later rotations //
      mLastTheta = mTheta;
      mLastPhi = mPhi;
      break;
    }
    case LEFT_BTN : {
      // left button has been pressed -> start new rotation -> save initial point //
      mX = x;
      mY = y;
      break;
    }
    case RIGHT_BTN : {
      // not used yet //
      break;
    }
    default : break;
  }
  mState = state;
}

void Trackball::updateOffset(Motion motion, float dist) {
  // init direction multiplicator (forward/backward, left/right are SYMMETRIC!) //
  int dir = 1;
  switch (motion) {
    case MOVE_FORWARD : {
      dir = -1;
    }
    case MOVE_BACKWARD : {
      // rotate unit vector (0,0,1) *looking up the z-axis* using theta and phi //
      // proceed in look direction by STEP_DISTANCE -> scale look vector and add to current view offset //
      // dir allows to reverse look vector //
      mViewOffset[0] += dir * dist * sin(mTheta) * cos(mPhi);
      mViewOffset[1] += dir * dist               * sin(mPhi);
      mViewOffset[2] += dir * dist * cos(mTheta) * cos(mPhi);
      break;
    }
    case MOVE_LEFT : {
      dir = -1;
    }
    case MOVE_RIGHT : {
      // drehmatrix um y
      // rotate unit vector (1,0,0) *looking up the x-axis* by theta only -> stay in x-z-plane //
      // add x and z components to current view offset //
      mViewOffset[0] += dir * dist * cos(mTheta);
      mViewOffset[2] += dir * dist * -sin(mTheta);
      break;
    }
    case MOVE_DOWN : {
      dir = -1;
    }
    case MOVE_UP : {
      if (mTheta != 0 || mPhi != 0)
        std::cout << "going up and down not correctly implemented!" << std::endl;
      mViewOffset[1] += dir * dist;
      break;
    }
    default : break;
  }
}

void Trackball::reset(float theta, float phi, float dist) { 
  mPhi = phi;
  mLastPhi = mPhi;
  mTheta = theta;
  mLastTheta = mTheta;
  mViewOffset[0] = sin(mTheta) * cos(mPhi) * dist;
  mViewOffset[1] =               sin(mPhi) * dist;
  mViewOffset[2] = cos(mTheta) * cos(mPhi) * dist;
  mX = 0;
  mY = 0;
  mState = NO_BTN;
}

void Trackball::rotateView(void) const {
  float x, y, z;
  std::tie(x, y, z) = getViewDirection();
  gluLookAt(mViewOffset[0], mViewOffset[1], mViewOffset[2], // from
            mViewOffset[0] + x, mViewOffset[1] + y, mViewOffset[2] + z,  // at
            0, 1, 0); // up
}

std::tuple<float, float, float> Trackball::getCameraPosition() const {
  return std::make_tuple(mViewOffset[0], mViewOffset[1], mViewOffset[2]);
}

std::tuple<float, float, float> Trackball::getViewDirection() const {
  // rotate view vector (0,0,1) *looking up the z-axis* according to theta and phi //
  float x = sin(mTheta) * cos(mPhi);
  float y =               sin(mPhi);
  float z = cos(mTheta) * cos(mPhi);
  return std::make_tuple(-x, -y, -z);
}
