#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


Mat get_mask(Mat img, Point center, int r){
    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    circle(mask, center, r, Scalar(255), -1);
    return mask;
}

Mat get_hist(Mat img_hvs, Point center, int r = 10, int bins = 18){
    Mat mask = get_mask(img_hvs, center, r);
    int hist_size[] = {bins};
    float hranges[] = {0, 180};
    const float* ranges[] = {hranges};
    int chanels [] = {0};
    MatND hist;
    calcHist(&img_hvs, 1, chanels, mask, hist, 1, hist_size, ranges);
    Scalar totoal = sum(hist);
    hist = hist * 1.0 / totoal[0];
    cout<<hist.size()<<endl;
    //cout<<format(hist, Formatter::FMT_PYTHON)<<endl;
    return hist;
}

double b_distance(Mat hist1, Mat hist2){
    double s = 0;
    for (int i=0; i<hist1.rows; i++){
        s += sqrt(hist1.at<double>(i, 0) * hist2.at<double>(i, 0));
    }
    return;
}




void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat* im = reinterpret_cast<Mat*>(param);
	switch (event)
	{
        
		case EVENT_LBUTTONDOWN:     //鼠标左键按下响应：返回坐标和灰度
			get_hist(*im, Point(x, y));
			break;
			break;			
	}
}
Mat drawMatch(Mat img1, Mat img2, bool extended = true){
   
    int minHessian=400;
    Ptr<SURF>  detector = SURF::create(minHessian, 4, 3, extended);
    vector<KeyPoint>key_points_img1, key_points_img2;
    detector->detect(img1, key_points_img1);
    detector->detect(img2, key_points_img2);
 
    
    Mat descriptor_img1, descriptor_img2;
    detector->compute(img1, key_points_img1, descriptor_img1);
    detector->compute(img2, key_points_img2, descriptor_img2);
    
    
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> good_matches;

    double ratio = 0.8;

    for(int i = 0; i < matches.size(); i++)
    {
        
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }
    
    sort(good_matches.begin(), good_matches.end());

    
    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
 
    
    for(unsigned int i = 0; i < min(30, (int)good_matches.size()); ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
    }

    Mat img_matches;

    drawMatches(img1, key_points_img1, img2, key_points_img2, good_matches, img_matches);
  
    Mat H = findHomography(img1_points, img2_points, RANSAC);
    
    vector<Point2f> img1_corners(4);
    img1_corners[0]=Point2f(0,0);
    img1_corners[1]=Point2f(img1.cols,0);
    img1_corners[2]=Point2f(img1.cols, img1.rows);
    img1_corners[3]=Point2f(0,img1.rows);
    
    vector<Point2f> img2_corners(4);
    cout<<format( H, Formatter::FMT_PYTHON)<<endl;
    if (!H.empty()){
        
        perspectiveTransform(img1_corners, img2_corners, H);
        
        line(img_matches, img2_corners[0]+Point2f(img1.cols,0), img2_corners[1] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_corners[1]+Point2f(img1.cols,0), img2_corners[2] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_corners[2]+Point2f(img1.cols,0), img2_corners[3] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_corners[3]+Point2f(img1.cols,0), img2_corners[0] + Point2f(img1.cols,0), Scalar(0, 255, 0), 2);

    }
       
    return img_matches;
  
}



int main()
{
    string folder = "/home/jiaming/research/img/";
    string img1_name = "pencil_bag.png";
    string img2_name = "large.png";
    Mat img1 = imread(folder + img1_name, IMREAD_COLOR);
    Mat img2 = imread(folder + img2_name, IMREAD_COLOR);
    cvtColor(img1, img1, COLOR_BGR2HSV_FULL);
    //Mat mask = get_mask(img1, Point(0, 0), 30);
    namedWindow("Hist", 0);
    setMouseCallback("Hist", onMouse, &img1);
    while (1)
    {
        imshow("Hist", img1);
        waitKey(40);
    }

    
    //cout<<img1.at<Vec3b>(0, 0)<<endl;
    //Mat out = drawMatch(img1, img2);
    //imshow("Matches", out);
    //waitKey(0);
    return 0;

}

