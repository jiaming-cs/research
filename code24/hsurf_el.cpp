// test frame 83 84


#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <map>
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"


using namespace std;
using namespace cv;


void drawMatch(Mat& img1, Mat& img2, Mat& img_matches, double ratio = 0.8, bool extended = false){
   
    int minHessian=100;
    SurfFeatureDetector  detector(minHessian, 4, 3, false);
    vector<KeyPoint>key_points_img1, key_points_img2;
    detector.detect(img1, key_points_img1);
    detector.detect(img2, key_points_img2);
 
    
    if(key_points_img1.size() < 4 || key_points_img2.size() < 4){
        return ;
    }

    Mat descriptor_img1, descriptor_img2;
    detector.compute(img1, key_points_img1, descriptor_img1);
    detector.compute(img2, key_points_img2, descriptor_img2);
    
    
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> good_matches;

    if(descriptor_img1.empty() || descriptor_img2.empty()){
        return ;
    }
    map<int, DMatch> matches_map;
   
    if (matches.size() < 30){
        ratio = 0.9;
    }

    for(int i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            if(matches_map.count(matches[i][0].trainIdx) == 0){
                matches_map[matches[i][0].trainIdx] = matches[i][0];
            }
            else{
                if (matches_map[matches[i][0].trainIdx].distance > matches[i][0].distance){
                    matches_map[matches[i][0].trainIdx] = matches[i][0];
                }
            }
        }
    }
      
    
    /*
    else
    {
        for(int i = 0; i < matches.size(); i++)
        {
            
            if(matches_map.count(matches[i][0].trainIdx) == 0){
                matches_map[matches[i][0].trainIdx] = matches[i][0];
            }
            else{
                if (matches_map[matches[i][0].trainIdx].distance > matches[i][0].distance){
                    matches_map[matches[i][0].trainIdx] = matches[i][0];
                }
            }
            
        }
       
    }
    */

    for (map<int, DMatch>::iterator it = matches_map.begin(); it != matches_map.end(); it++){
        good_matches.push_back(it->second);
    }

    if (good_matches.size() < 4){
        return ;
    }

    
    sort(good_matches.begin(), good_matches.end());


    
    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
    
    int num_good = min((int)good_matches.size(), 50);
    vector<DMatch> first_matches;
    
    for(unsigned int i = 0; i < num_good; ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
        first_matches.push_back(good_matches[i]);
    }

    

   

    drawMatches(img1, key_points_img1, img2, key_points_img2, first_matches, img_matches);
    
    cv::putText(img_matches, to_string(key_points_img1.size()) + "   " + to_string(key_points_img2.size()), Point(0, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0) );
    Mat H = findHomography(img1_points, img2_points, RANSAC, 3);
    
    vector<Point2f> img1_cross(5);
    Point2f middle = Point2f(img1.cols/2, img1.rows/2);
    img1_cross[0]=middle+Point2f(0,-50);
    img1_cross[1]=middle+Point2f(0, 50);
    img1_cross[2]=middle+Point2f(-50, 0);
    img1_cross[3]=middle+Point2f(50, 0);
    img1_cross[4]=middle;
    vector<Point2f> img2_cross(5);
   
    if (!H.empty()){
        
        perspectiveTransform(img1_cross, img2_cross, H);

        line(img_matches, img1_cross[0], img1_cross[1], Scalar(0, 255, 0), 2);
        line(img_matches, img1_cross[2], img1_cross[3], Scalar(0, 255, 0), 2);
        
        line(img_matches, img2_cross[0]+Point2f(img1.cols, 0), img2_cross[1]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_cross[2]+Point2f(img1.cols, 0), img2_cross[3]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        circle(img_matches, img2_cross[4]+Point2f(img1.cols, 0), 5, Scalar(0, 0, 255), 2);
    }
       
    return ;
  
}


double* get_hist(Mat img_hsv, Point center, int r = 10, int bins = 60){
    
    Mat g_kernal_x = getGaussianKernel(2*r, 2);
    Mat g_kernal_y = getGaussianKernel(2*r, 2);
    Mat g_kernal_t = g_kernal_x * g_kernal_y.t();
    Mat g_kernal = g_kernal_t(Rect(r, r, r, r));
    normalize(g_kernal, g_kernal, 1, 100, NORM_MINMAX);
    g_kernal.convertTo(g_kernal, CV_8UC1);
    double span = 360 / bins;
    double* temp = new double[bins]();
    int i_start = max((int)center.y - r, 0);
    int i_end = min((int)center.y + r, (int)img_hsv.rows);
    int j_start = max((int)center.x - r, 0);
    int j_end = min((int)center.x + r, (int)img_hsv.cols);
    for (int i = i_start; i < i_end; i++){
        for (int j = j_start; j < j_end; j++){
            int dy = abs(i - center.y);
            int dx = abs(j - center.x);
            if ( dx*dx + dy*dy > r*r)
                continue;
            int h = img_hsv.at<uchar>(i, j);
            temp[(int)(h / span)] += (int)g_kernal.at<uchar>(dy, dx);
        }
    }
    
    
    int total = std::accumulate(temp, temp+bins, 0);
    for (int i=0; i<bins; i++){
        temp[i] /= total;
    }
    return temp;
}


double b_distance(double* hist1, double* hist2, int bins){
    double s = 0;
    for (int i=0; i<bins; i++){
        s += sqrt(hist1[i] * hist2[i]);
    }
    delete[] hist1;
    delete[] hist2;
    return s;
}

double get_distance(Mat img1_h, Mat img2_h, KeyPoint kp1, KeyPoint kp2, int bins = 60){
   
    
    double* h1 = get_hist(img1_h, kp1.pt, kp1.size, bins);
    double* h2 = get_hist(img2_h, kp2.pt, kp2.size, bins);

    return b_distance(h1, h2, bins);
}


Mat get_hue(Mat img){
    vector<Mat> chanles;
    split(img, chanles);
    Mat out = chanles[0].clone();
    return out;
}


void drawMatch_hsurf(Mat& img1, Mat& img2, Mat& img_matches, double ratio = 0.8, double weight=0.5, bool extended = false){
   
    int minHessian=100;
    SurfFeatureDetector  detector(minHessian, 4, 3, false);
    vector<KeyPoint>key_points_img1, key_points_img2;
    detector.detect(img1, key_points_img1);
    detector.detect(img2, key_points_img2);
 
    
    if(key_points_img1.size() < 4 || key_points_img2.size() < 4){
        return ;
    }

    Mat descriptor_img1, descriptor_img2;
    detector.compute(img1, key_points_img1, descriptor_img1);
    detector.compute(img2, key_points_img2, descriptor_img2);
    
    
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> good_matches;

    if(descriptor_img1.empty() || descriptor_img2.empty()){
        return ;
    }
    map<int, DMatch> matches_map;
   
    if (matches.size() < 30){
        ratio = 0.9;
    }
    Mat img1_h = get_hue(img1);
    Mat img2_h = get_hue(img2);
    for(int i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            if(matches_map.count(matches[i][0].trainIdx) == 0){
                int query_index = matches[i][0].queryIdx;
                int train_index = matches[i][0].trainIdx;
                double hsv_distance = get_distance(img1_h, img2_h, key_points_img1[query_index], key_points_img2[train_index]) ;
                double t = matches[i][0].distance;
                matches[i][0].distance = (1 - weight) * t + weight * (1-hsv_distance) * t;
                matches_map[matches[i][0].trainIdx] = matches[i][0];
            }
            else{
                int query_index = matches[i][0].queryIdx;
                int train_index = matches[i][0].trainIdx;
                double hsv_distance = get_distance(img1_h, img2_h, key_points_img1[query_index], key_points_img2[train_index]) ;
                double t = matches[i][0].distance;
                matches[i][0].distance = (1 - weight) * t + weight * (1-hsv_distance) * t;
                if (matches_map[matches[i][0].trainIdx].distance > matches[i][0].distance){
                    matches_map[matches[i][0].trainIdx] = matches[i][0];
                }
            }
        }
    }
      
    
    /*
    else
    {
        for(int i = 0; i < matches.size(); i++)
        {
            
            if(matches_map.count(matches[i][0].trainIdx) == 0){
                matches_map[matches[i][0].trainIdx] = matches[i][0];
            }
            else{
                if (matches_map[matches[i][0].trainIdx].distance > matches[i][0].distance){
                    matches_map[matches[i][0].trainIdx] = matches[i][0];
                }
            }
            
        }
       
    }
    */

    for (map<int, DMatch>::iterator it = matches_map.begin(); it != matches_map.end(); it++){
        good_matches.push_back(it->second);
    }

    if (good_matches.size() < 4){
        return ;
    }

    
    sort(good_matches.begin(), good_matches.end());


    
    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
    
    int num_good = min((int)good_matches.size(), 50);
    vector<DMatch> first_matches;
    
    for(unsigned int i = 0; i < num_good; ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
        first_matches.push_back(good_matches[i]);
    }

    

   

    drawMatches(img1, key_points_img1, img2, key_points_img2, first_matches, img_matches);
    
    cv::putText(img_matches, to_string(key_points_img1.size()) + "   " + to_string(key_points_img2.size()), Point(0, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0) );
    Mat H = findHomography(img1_points, img2_points, LMEDS, 3);
    
    vector<Point2f> img1_cross(5);
    Point2f middle = Point2f(img1.cols/2, img1.rows/2);
    img1_cross[0]=middle+Point2f(0,-50);
    img1_cross[1]=middle+Point2f(0, 50);
    img1_cross[2]=middle+Point2f(-50, 0);
    img1_cross[3]=middle+Point2f(50, 0);
    img1_cross[4]=middle;
    vector<Point2f> img2_cross(5);
   
    if (!H.empty()){
        
        perspectiveTransform(img1_cross, img2_cross, H);

        line(img_matches, img1_cross[0], img1_cross[1], Scalar(0, 255, 0), 2);
        line(img_matches, img1_cross[2], img1_cross[3], Scalar(0, 255, 0), 2);
        
        line(img_matches, img2_cross[0]+Point2f(img1.cols, 0), img2_cross[1]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_cross[2]+Point2f(img1.cols, 0), img2_cross[3]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        circle(img_matches, img2_cross[4]+Point2f(img1.cols, 0), 5, Scalar(0, 0, 255), 2);
    }
       
    return ;
  
}

int main(int argc, char const *argv[])
{
    string comp_folder = "/home/jiaming/opencv24/code24/out_frames/";
    string folder = "/home/jiaming/opencv24/code24/img";
    string img1_name = "1/";
    string img2_name = "2/";
    string extention = ".png";
    int i=0;
    Mat img_org;
    Mat img_hsurf;
    while (i<115)
    {
    Mat img1 = imread(folder + img1_name + to_string(i) + extention, IMREAD_COLOR);
    Mat img2 = imread(folder + img2_name + to_string(i) + extention, IMREAD_COLOR);
    //Mat org = imread(comp_folder + to_string(i) + extention, IMREAD_COLOR);
    drawMatch(img1, img2, img_org);
    if (!img_org.empty())
        imshow("org", img_org);
    drawMatch_hsurf(img1, img2, img_hsurf);
    if (!img_hsurf.empty())
        imshow("Hsurf", img_hsurf);
    //drawMatch(img2, img1, reverse);
    //if (!reverse.empty())
        //imshow("Reverse", reverse);
    
    waitKey( 0 );
    i++;
    }
    
   

    return 0;
}

