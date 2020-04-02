#include <opencv2/core.hpp> //types
#include <opencv2/imgcodecs.hpp> // read image
#include <opencv2/objdetect.hpp> // for cascade classifier

#include <opencv2/highgui.hpp> // display image in window
#include <opencv2/imgproc.hpp> //draw

#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

const string FACEMODEL = "/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
const string EYESMODEL = "/opt/opencv/data/haarcascades/haarcascade_eye.xml";

int main(int argc, char **argv)
{
	Mat finalImage;
	// There is no arg in the excecution
	if (argc != 2) { cout << " There is no image to load \n"; return -1; }

	// Image from input
	Mat image;
	image = imread(argv[1], IMREAD_GRAYSCALE); // Read the file
	if (image.empty()) { cout << "Could not open or find the image \n"; return -1; }

	// PreDefined trained XML classifiers with facial features
	CascadeClassifier face_cascade, eyes_cascade;

	// Load classifiers
	if (!face_cascade.load(FACEMODEL)) { cout << "Error loading face cascade\n"; return -1; }
	if (!eyes_cascade.load(EYESMODEL)) { cout << "Error loading eyes cascade\n"; return -1; }


	// Detect faces
	vector<Rect> faces;
	face_cascade.detectMultiScale( image, faces );

	cout << "Faces found: " << faces.size() << "\n\n";
	for ( size_t i = 0; i < faces.size(); i++ )
	{
		cout << "Face[" << i + 1 << "] | " << "ROI: " << faces[i] << "\n";

		Mat faceROI = image( faces[i] );
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);
		cout << "Eyes found: " << eyes.size() << "\n";

		if(eyes.size() == 2) {
				// Eye A
				cout << "Eye A: " << eyes[0] << "\n";
				Point eye_center_A( faces[i].x + eyes[0].x + eyes[0].width/2, faces[i].y + eyes[0].y + eyes[0].height/2 );
				// Eye B
				cout << "Eye B: " << eyes[1] << "\n";
				Point eye_center_B( faces[i].x + eyes[1].x + eyes[1].width/2, faces[i].y + eyes[1].y + eyes[1].height/2 );

				//get angle | JUST GET THE POSITIVE
				// https://math.stackexchange.com/questions/1201337/finding-the-angle-between-two-points
				double angleA = atan2(eye_center_A.y - eye_center_B.y, eye_center_A.x - eye_center_B.x) * 180 / M_PI;
				cout << "Angle A: " << angleA << "\n";
				double angleB = atan2(eye_center_B.y - eye_center_A.y, eye_center_B.x - eye_center_A.x) * 180 / M_PI;
				cout << "Angle B: " << angleB << "\n";

				//rotate
				Point face_center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height );
				Mat rotationMatrix = getRotationMatrix2D(face_center, angleA > 0 ? angleA : angleB , 1.0);
				warpAffine(image(faces[i]), finalImage, rotationMatrix, faces[i].size());

			//size(x,y)
			//resize(image, finalImage, Size(200,200)); 

			imwrite("img1.jpg", finalImage);

			cout << "\n";
		} else {
			cout << "No se procesa" << "\n";
		}
		//draw
		/*for ( size_t j = 0; j < eyes.size(); j++ )
		{
			Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
			int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
			circle( image, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
		}*/
	}

	//show images
	namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
	imshow( "Display window", finalImage ); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}