#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char const *argv[])
{
    VideoCapture cp;
    cp.open("/home/jiaming/Documents/research/videos/Robot2.mp4");
    if (!cp.isOpened()){
        cout << "Fail to open the video!"<< endl;
    }
    int i = 1;
    int index = 1;
    Mat frame;
    cp >> frame;
    while (!frame.empty()&i<201){
        if (i < 100){
            i ++;
            continue;
        }
        resize(frame, frame, Size(500, 300));
        imwrite("/home/jiaming/Documents/research/img/v2/" + to_string(index) + ".png", frame);
        i++;
        index++;
        cp >> frame;
    }
    return 0;
}
