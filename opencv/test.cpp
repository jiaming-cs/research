#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace cv::xfeatures2d;

Mat src;
int minHessian = 50;
void trackBar(int, void*);
int main()
{
    src = imread("large.jpg");
    if (src.empty())
    {
        printf("can not load image \n");
        return -1;
    }
    namedWindow("input", WINDOW_AUTOSIZE);
    imshow("input", src);

    namedWindow("output", WINDOW_AUTOSIZE);
    createTrackbar("minHessian","output",&minHessian, 500, trackBar);

    waitKey(0);
    return 0;
}


void trackBar(int, void*)
{
    Mat dst;
    // SURF特征检测
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints;
    detector->detect(src, keypoints, Mat());
    // 绘制关键点
    drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("output", dst);
}
