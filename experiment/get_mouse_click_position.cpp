#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp>

#include <string>
using namespace cv;
using namespace std;
void draw_rectangle(int event, int x, int y, int flags, void*);

Point2f p;
int main(int argc, char *argv[])
{
    
    Mat frame;
    string img_name = "/home/jiaming/Documents/research/img/v2/1.png";
    frame = imread(img_name, IMREAD_COLOR);
    
   
    while(true){
        imshow("output", frame);
        setMouseCallback("output", draw_rectangle, 0);
        circle(frame , p, 5, Scalar(0, 255, 0), 0);
        waitKey(30);

    }
    
    
    return 0;
}

//框选目标
void draw_rectangle(int event, int x, int y, int flags, void*)
{
    if (event == EVENT_LBUTTONDOWN)
        {
            cout<< x << ", " << y <<endl;
            p = Point2f(x, y);

        }
    
}
