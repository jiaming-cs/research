#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp> // VideoCapture
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
    
    VideoCapture vc = VideoCapture("/home/jiaming/research/videos/test.mp4");
    int index = 0;
    int f = 15;
    int num = 0;
    Mat frame;
    while(vc.read(frame)){
        if (index % f == 0){
            imwrite("/home/jiaming/research/img/2fps/" + to_string(num) + ".png", frame);
            num++;
        }
        index++;
    }
    
    return 0;
}
