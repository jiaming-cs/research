// test frame 83 84


#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <map>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

double get_distance(const Point2f &p1, const Point2f& p2){
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}


vector<Point2f> draw_match(Mat& img1, Mat& img2, Mat& img_matches, Ptr<SURF>& detector, bool extended = false){
   
    
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
    
    
    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
 
    
    for(unsigned int i = 0; i < (int)good_matches.size(); ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
    }

   

    drawMatches(img1, key_points_img1, img2, key_points_img2, good_matches, img_matches);
  
    Mat H = findHomography(img1_points, img2_points, RHO);
    
    vector<Point2f> img1_cross(10);
    img1_cross[0]=Point2f(213, 153);
    img1_cross[1]=Point2f(262, 154);
    img1_cross[2]=Point2f(227, 175);
    img1_cross[3]=Point2f(259, 176);
    img1_cross[4]=Point2f(188, 181);
    img1_cross[5]=Point2f(289, 202);
    img1_cross[6]=Point2f(189, 235);
    img1_cross[7]=Point2f(291, 257);
    img1_cross[8]=Point2f(201, 272);
    img1_cross[9]=Point2f(339, 231);
    
   

    vector<Point2f> img2_cross(10);
    
    for (int i=0; i< 10; i++){

        img2_cross[i] = Point2f(0, 0);
    }
   
    if (!H.empty()){
        
        perspectiveTransform(img1_cross, img2_cross, H);

    }
       
    return img2_cross ;
  
  
}


void func(int min_size){

    string folder = "/home/jiaming/Documents/research/img/";
    string img1_name = "v1/";
    string img2_name = "v2/";
    string extention = ".jpg";
    int i=1;
    Mat out;

    Ptr<SURF> detector = SURF::create(800);
    
    vector<Point2f> central_point;
    time_t start, end;
    start = clock();
    Point2f p_zero = Point2f(0, 0);
    while (i<101)
    {

    Mat img1 = imread(folder + img1_name + to_string(i) + extention, IMREAD_COLOR);
    Mat img2 = imread(folder + img2_name + to_string(i) + extention, IMREAD_COLOR);
    Point2f p = draw_match(img1, img2, out, detector);

    central_point.push_back(p);
     

    i++;
    }



}


int main(int argc, char const *argv[])
{

    for(int i = 25; i<=500; i+=25){

        func(i);

    }
   

    return 0;

}

