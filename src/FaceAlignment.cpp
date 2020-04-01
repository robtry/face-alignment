#include "FaceAlignment.hpp"
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

const string FACEMODEL;
const string EYESMODEL;
void detectEyes(Mat face);
Point getEyeCenter(const Mat &face, const Mat &eye);
Point getFaceCenter(const Mat &face);
double getAngleBetweenEyes(const Mat &eyeA, const Mat &eyeB);
void rotate(Mat &face);
void crop(const int height, const int width);