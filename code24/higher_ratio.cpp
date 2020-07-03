// test frame 83 84


#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <map>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

bool compare_hessian(const KeyPoint &kp1, const KeyPoint &kp2) {
    return kp1.response > kp2.response;
}


void draw_match_double_surf(Mat& img1,
                Mat& img2,
                Mat& img_matches,
                Ptr<SurfFeatureDetector> &detector1,
                Ptr<SurfFeatureDetector> &detector2,
                int max_hessian = 800,
                int min_hessian = 5,
                int max_size = 300,
                int min_size = 100,
                double ratio = 0.8, 
                bool extended = false){
    
                    
    vector<KeyPoint>key_points_img1, key_points_img2;
    time_t start, end;
 
    detector1->detect(img1, key_points_img1);

    int hessian = detector1->hessianThreshold;
    //cout << "Threshold_start:" << detector->hessianThreshold <<endl;

    while ((key_points_img1.size() < min_size || key_points_img1.size() > max_size) && hessian >= min_hessian){
        if ( key_points_img1.size() < min_size ){
            hessian = (int)hessian/1.5;
            detector1.release();
            //start = clock();
            detector1 = new SurfFeatureDetector(hessian);
            //end = clock();
            //cout<<"Time for new a SURF objct: "<< (end-start) * 1.0 / CLOCKS_PER_SEC <<endl;
            detector1->detect(img1, key_points_img1);
        }

        else
        {
            sort(key_points_img1.begin(), key_points_img1.end(), compare_hessian);
            key_points_img1.resize(max_size);
        }
        
    }
    cout<<"Hessian1: "<<detector1->hessianThreshold<<endl;

    detector2->detect(img2, key_points_img2);

    hessian = detector2->hessianThreshold;

    while ((key_points_img2.size() < min_size || key_points_img2.size() > max_size) && hessian >= min_hessian){
        if ( key_points_img2.size() < min_size ){
            hessian = (int)hessian/1.5;
            detector2.release();
            //start = clock();
            detector2 = new SurfFeatureDetector(hessian);
            //end = clock();
            //cout<<"Time for new a SURF objct: "<< (end-start) * 1.0 / CLOCKS_PER_SEC <<endl;
            detector2->detect(img2, key_points_img2);
        }

        else
        {
            sort(key_points_img2.begin(), key_points_img2.end(), compare_hessian);
            key_points_img2.resize(max_size);
        }
        
    }
    cout<<"Hessian2: "<<detector2->hessianThreshold<<endl;
    cout<<"Size: "<<key_points_img2.size()<<endl;
    
    //cout << "Threshold_end:" << detector->hessianThreshold <<endl;
    if(key_points_img1.size() < 4 || key_points_img2.size() < 4){
        return ;
    }

    Mat descriptor_img1, descriptor_img2;
    detector1->compute(img1, key_points_img1, descriptor_img1);
    detector2->compute(img2, key_points_img2, descriptor_img2);
    

    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> good_matches;

    if(descriptor_img1.empty() || descriptor_img2.empty()){
        return ;
    }
    
    map<int, DMatch> matches_map;
   
    //double ratio = max_ratio  - (max_ratio - min_ratio) * ((key_points_img1.size() - min_size) * 1.0 / (max_size - min_size));
    //cout<<"ratio: "<<ratio<<endl;

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
      

    for (map<int, DMatch>::iterator it = matches_map.begin(); it != matches_map.end(); it++){
        good_matches.push_back(it->second);
    }

    if (good_matches.size() < 4){
        return ;
    }
    cout<<"Number of vaild point: "<< good_matches.size()<<endl;
    
    sort(good_matches.begin(), good_matches.end());


    
    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
    
    int num_good = min((int)good_matches.size(), (int)good_matches.size());
    vector<DMatch> first_matches;
    
    for(unsigned int i = 0; i < num_good; ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
        first_matches.push_back(good_matches[i]);
    }

    

   

    drawMatches(img1, key_points_img1, img2, key_points_img2, first_matches, img_matches);
    
    cv::putText(img_matches, to_string(key_points_img1.size()) + "   " + to_string(key_points_img2.size()), Point(0, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0) );
    Mat H = findHomography(img1_points, img2_points, RANSAC, 10);
    
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

void draw_match_double_surf_reduce(Mat& img1,
                Mat& img2,
                Mat& img_matches,
                Ptr<SurfFeatureDetector> &detector1,
                Ptr<SurfFeatureDetector> &detector2,
                int max_hessian = 800,
                int min_hessian = 5,
                int max_size = 300,
                int min_size = 100,
                double ratio = 0.95, 
                double reduce = 0.8,
                bool extended = false){
    
                    
    vector<KeyPoint>key_points_img1, key_points_img2;
    time_t start, end;
 
    detector1->detect(img1, key_points_img1);

    int hessian = detector1->hessianThreshold;
    //cout << "Threshold_start:" << detector->hessianThreshold <<endl;

    while ((key_points_img1.size() < min_size || key_points_img1.size() > max_size) && hessian >= min_hessian){
        if ( key_points_img1.size() < min_size ){
            hessian = (int)hessian/1.5;
            detector1.release();
            //start = clock();
            detector1 = new SurfFeatureDetector(hessian);
            //end = clock();
            //cout<<"Time for new a SURF objct: "<< (end-start) * 1.0 / CLOCKS_PER_SEC <<endl;
            detector1->detect(img1, key_points_img1);
        }

        else
        {
            sort(key_points_img1.begin(), key_points_img1.end(), compare_hessian);
            key_points_img1.resize(max_size);
        }
        
    }
    cout<<"Hessian1: "<<detector1->hessianThreshold<<endl;

    detector2->detect(img2, key_points_img2);

    hessian = detector2->hessianThreshold;

    while ((key_points_img2.size() < min_size || key_points_img2.size() > max_size) && hessian >= min_hessian){
        if ( key_points_img2.size() < min_size ){
            hessian = (int)hessian/1.5;
            detector2.release();
            //start = clock();
            detector2 = new SurfFeatureDetector(hessian);
            //end = clock();
            //cout<<"Time for new a SURF objct: "<< (end-start) * 1.0 / CLOCKS_PER_SEC <<endl;
            detector2->detect(img2, key_points_img2);
        }

        else
        {
            sort(key_points_img2.begin(), key_points_img2.end(), compare_hessian);
            key_points_img2.resize(max_size);
        }
        
    }
    cout<<"Hessian2: "<<detector2->hessianThreshold<<endl;
    cout<<"Size: "<<key_points_img2.size()<<endl;
    
    //cout << "Threshold_end:" << detector->hessianThreshold <<endl;
    if(key_points_img1.size() < 4 || key_points_img2.size() < 4){
        return ;
    }

    Mat descriptor_img1, descriptor_img2;
    detector1->compute(img1, key_points_img1, descriptor_img1);
    detector2->compute(img2, key_points_img2, descriptor_img2);
    

    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> good_matches;

    if(descriptor_img1.empty() || descriptor_img2.empty()){
        return ;
    }
    
    map<int, DMatch> matches_map;
   
    //double ratio = max_ratio  - (max_ratio - min_ratio) * ((key_points_img1.size() - min_size) * 1.0 / (max_size - min_size));
    //cout<<"ratio: "<<ratio<<endl;

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
      

    for (map<int, DMatch>::iterator it = matches_map.begin(); it != matches_map.end(); it++){
        good_matches.push_back(it->second);
    }

    if (good_matches.size() < 4){
        return ;
    }
    cout<<"Number of vaild point: "<< good_matches.size()<<endl;
    
    sort(good_matches.begin(), good_matches.end());

    //good_matches.resize((int)(good_matches.size()*reduce));


    
    vector<Point2f> img1_points;
    vector<Point2f> img2_points;
    
    int num_good = min((int)good_matches.size(), (int)good_matches.size());
    vector<DMatch> first_matches;
    
    for(unsigned int i = 0; i < num_good; ++i)
    {
        img1_points.push_back(key_points_img1[good_matches[i].queryIdx].pt);
        img2_points.push_back(key_points_img2[good_matches[i].trainIdx].pt);
        first_matches.push_back(good_matches[i]);
    }

    

   

    drawMatches(img1, key_points_img1, img2, key_points_img2, first_matches, img_matches);
    
    cv::putText(img_matches, to_string(key_points_img1.size()) + "   " + to_string(key_points_img2.size()), Point(0, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0) );
    Mat H = findHomography(img1_points, img2_points, RANSAC, 10);
    
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
    Mat out1, out2;

    Ptr<SurfFeatureDetector> detector_a = new SurfFeatureDetector(200);
    Ptr<SurfFeatureDetector> detector_b = new SurfFeatureDetector(200);

    Ptr<SurfFeatureDetector> detector1 = new SurfFeatureDetector(200);
    Ptr<SurfFeatureDetector> detector2 = new SurfFeatureDetector(200);
    while (i<115)
    {

    Mat img1 = imread(folder + img1_name + to_string(i) + extention, IMREAD_COLOR);
    Mat img2 = imread(folder + img2_name + to_string(i) + extention, IMREAD_COLOR);
 
    draw_match_double_surf_reduce(img1, img2, out1, detector_a, detector_b);
    if (!out1.empty())
        imshow("0.95", out1);

    draw_match_double_surf(img1, img2, out2, detector1, detector2);
    if (!out2.empty())
        imshow("0.8", out2);

    waitKey(0);

    i++;
    }
    
   

    return 0;
}

