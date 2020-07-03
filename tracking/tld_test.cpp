#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <string>
using namespace cv;
using namespace std;
void draw_rectangle(int event, int x, int y, int flags, void*);
Mat firstFrame;
Point previousPoint, currentPoint;
Rect2d bbox;

string to_frame_name(int i){
    string temp = to_string(i);
    if (temp.size() == 1)
        return "000" + temp;
    else if(temp.size() == 2)
        return "00" + temp;
    else 
        return "0" + temp;
}

int main(int argc, char *argv[])
{
    
    Mat frame;
    string foder = "/home/jiaming/Documents/research/img/Jogging/img/";
    firstFrame = imread(foder + to_frame_name(1) + ".jpg", IMREAD_COLOR);
    
   
    if(!firstFrame.empty())
    {
        namedWindow("output", WINDOW_AUTOSIZE);
        imshow("output", firstFrame);
        setMouseCallback("output", draw_rectangle, 0);
        waitKey();
    }
    //使用TrackerMIL跟踪
    //Ptr<TrackerMIL> tracker= TrackerMIL::create();
    Ptr<TrackerTLD> tracker= TrackerTLD::create();
    //Ptr<TrackerKCF> tracker = TrackerKCF::create();
    //Ptr<TrackerMedianFlow> tracker = TrackerMedianFlow::create();
    //Ptr<TrackerBoosting> tracker= TrackerBoosting::create();
    int start_index = 2;
    int end_index = 307;
    tracker->init(frame,bbox);
    namedWindow("output", WINDOW_AUTOSIZE);
    for (int i = start_index; i <= end_index; i++){
        frame = imread(foder + to_frame_name(i) + ".jpg", IMREAD_COLOR);
        tracker->update(frame,bbox);
        rectangle(frame,bbox, Scalar(255, 0, 0), 2, 1);
        imshow("output", frame);
        if(waitKey(20)=='q')
        return 0;

    }
   
    destroyWindow("output");
    return 0;
}

//框选目标
void draw_rectangle(int event, int x, int y, int flags, void*)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        previousPoint = Point(x, y);
    }
    else if (event == EVENT_MOUSEMOVE && (flags&EVENT_FLAG_LBUTTON))
    {
        Mat tmp;
        firstFrame.copyTo(tmp);
        currentPoint = Point(x, y);
        rectangle(tmp, previousPoint, currentPoint, Scalar(0, 255, 0, 0), 1, 8, 0);
        imshow("output", tmp);
    }
    else if (event == EVENT_LBUTTONUP)
    {
        bbox.x = previousPoint.x;
        bbox.y = previousPoint.y;
        bbox.width = abs(previousPoint.x-currentPoint.x);
        bbox.height =  abs(previousPoint.y-currentPoint.y);
    }
    else if (event == EVENT_RBUTTONUP)
    {
        destroyWindow("output");
    }
}
