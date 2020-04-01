#include "module1.hpp"
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

string Module1::readImage(const string file){
	cout << "getting" << file <<"\n";
	return file;
}

void Module1::detectFaces(Mat image){
	readImage("qui onda");
}

