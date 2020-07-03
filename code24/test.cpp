#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
int main(int argc, char** argv)
{
    /*
	Mat image;
	image = imread("/home/odroid/Videos/large.png", 1);
	if (!image.data)
	{
		printf("No image data \n");
		return -2;
	}
    */

	cv::VideoCapture oVideoCapture;
//	bool bReturn = oVideoCapture.open(0);
	bool bReturn = oVideoCapture.open("/home/jiaming/opencv24/stabal.mp4");
//	bool bReturn = oVideoCapture.open("/home/odroid/Videos/RHF.mp4");

	namedWindow("Display Image", WINDOW_AUTOSIZE);
	//imshow("Display Image", image);
	waitKey(0);
	return 0;
}