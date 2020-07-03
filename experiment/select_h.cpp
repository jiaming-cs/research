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

bool compare_hessian(const KeyPoint &kp1, const KeyPoint &kp2) {
    return kp1.response > kp2.response;
}

bool compare_ratio(const pair<DMatch, double> &p1, const pair<DMatch, double> &p2){
    return p1.second < p2.second;
}

Point2f get_position(const Point2f &p, const Mat& affineMat){
    double x = p.x * affineMat.at<double>(0, 0) + p.y * affineMat.at<double>(0, 1) + affineMat.at<double>(0, 2);
    double y = p.x * affineMat.at<double>(1, 0) + p.y * affineMat.at<double>(1, 1) + affineMat.at<double>(1, 2);
    return Point2f(x, y);
}

double get_distance(const Point2f &p1, const Point2f& p2){
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

Mat get_better_h(const vector<Point2f> &img1_points, const vector<Point2f> &img2_points, Mat h1, Mat h2){
    double error1 = 0;
    double error2 = 0;
    Point2f p1, p2;

    for(int i = 0 ; i < img1_points.size(); i++){
        p1 = get_position(img1_points[i], h1);
        p2 = get_position(img1_points[i], h2);
        error1 += get_distance(p1, img2_points[i]);
        error2 += get_distance(p2, img2_points[i]);
    }
    
    //cout<<"error1:  "<<error1<<endl<<"error2:  "<<error2<<endl;
    if (error1 > error2){
        //cout<<"LEMED"<<endl;
        return h2;
    }
        
    else{
        //cout<<"RANSAC"<<endl;
        return h1;
    }
        
     
}


Point2f draw_match_double(Mat& img1,
                Mat& img2,
                Mat& img_matches,
                Ptr<SURF> &detector1,
                Ptr<SURF> &detector2,
                 int max_size = 300,
                int min_size = 200,
                int max_hessian = 800,
                int min_hessian = 5,
                int good_match_num = 50, 
                bool extended = false){
    
                    
    vector<KeyPoint>key_points_img1, key_points_img2;
    time_t start, end;
 
    detector1->detect(img1, key_points_img1);

    int hessian = detector1->getHessianThreshold();


    start = clock();
    while ((key_points_img1.size() < min_size || key_points_img1.size() > max_size) ){
        if ( key_points_img1.size() < min_size ){
            if (hessian <= min_hessian)
                break;
            hessian = (int)(hessian/1.5);
            detector1->setHessianThreshold(hessian);
            detector1->detect(img1, key_points_img1);
        }
        else
        {
            sort(key_points_img1.begin(), key_points_img1.end(), compare_hessian);
            key_points_img1.resize(max_size);
            hessian = (int)(hessian * 1.5 );  
            detector1->setHessianThreshold(hessian);
            break;
        }
        
    }
  

    detector2->detect(img2, key_points_img2);

    hessian = detector2->getHessianThreshold();

    while ((key_points_img2.size() < min_size || key_points_img2.size() > max_size) && hessian >= min_hessian){
        if ( key_points_img2.size() < min_size ){
            if (hessian <= min_hessian)
                break;
            hessian = (int)(hessian/1.5);
            detector2->setHessianThreshold(hessian);
            detector2->detect(img2, key_points_img2);
        }

        else
        {
            //sort(key_points_img2.begin(), key_points_img2.end(), compare_hessian);
            key_points_img2.resize(max_size);
            hessian = (int)(hessian * 1.5);
            detector2->setHessianThreshold(hessian);
            break;
        }
        
    }



    Mat descriptor_img1, descriptor_img2;

    detector1->compute(img1, key_points_img1, descriptor_img1);
    detector2->compute(img2, key_points_img2, descriptor_img2);
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
   
    vector<DMatch> good_matches;

   
    
    
    map<int, pair<DMatch, double>> matches_map;

   

    for(int i = 0; i < matches.size(); i++)
    {
        double ratio = matches[i][0].distance / matches[i][1].distance;
        if(matches_map.count(matches[i][0].trainIdx) == 0){

            pair<DMatch, double> p(matches[i][0], ratio);
            matches_map[matches[i][0].trainIdx] = p;
        }

        else{

            if (matches_map[matches[i][0].trainIdx].first.distance > matches[i][0].distance){
                pair<DMatch, double> p(matches[i][0], ratio);
                matches_map[matches[i][0].trainIdx] = p;
            }
        }
    }

    vector<pair<DMatch, double>> matches_and_ratio;

    for (map<int, pair<DMatch, double>>::iterator it =matches_map.begin(); it != matches_map.end(); it++){
        matches_and_ratio.push_back(it->second);
    }     

    sort(matches_and_ratio.begin(), matches_and_ratio.end(), compare_ratio);

    if (good_match_num < matches_and_ratio.size()){
        matches_and_ratio.resize(good_match_num);
    }
 

    for (vector<pair<DMatch, double>>::iterator it = matches_and_ratio.begin(); it != matches_and_ratio.end(); it++){
        good_matches.push_back(it->first);
    }



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

    Mat H1 = findHomography(img1_points, img2_points, RANSAC, 3);
    Mat H2 = findHomography(img1_points, img2_points, RHO, 8);
    Mat H3 = findHomography(img1_points, img2_points, RHO, 5);
    Mat H = get_better_h(img1_points, img2_points, H1, H2);
    H = get_better_h(img1_points, img2_points, H, H3);

    vector<Point2f> img1_cross(5);
    Point2f middle = Point2f(img1.cols/2, img1.rows/2);
    img1_cross[0]=middle+Point2f(0,-50);
    img1_cross[1]=middle+Point2f(0, 50);
    img1_cross[2]=middle+Point2f(-50, 0);
    img1_cross[3]=middle+Point2f(50, 0);
    img1_cross[4]=Point(624, 192);
    vector<Point2f> img2_cross(5);
   
    if (!H.empty()){
        
        perspectiveTransform(img1_cross, img2_cross, H);

        line(img_matches, img1_cross[0], img1_cross[1], Scalar(0, 255, 0), 2);
        line(img_matches, img1_cross[2], img1_cross[3], Scalar(0, 255, 0), 2);
        
        line(img_matches, img2_cross[0]+Point2f(img1.cols, 0), img2_cross[1]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        line(img_matches, img2_cross[2]+Point2f(img1.cols, 0), img2_cross[3]+Point2f(img1.cols, 0), Scalar(0, 255, 0), 2);
        circle(img_matches, img2_cross[4]+Point2f(img1.cols, 0), 5, Scalar(0, 0, 255), 2);
    }

    return img2_cross[4];
  
}


void func(int min_size){

    string folder = "/home/jiaming/Documents/research/code24/img";
    string img1_name = "1/";
    string img2_name = "2/";
    string extention = ".png";
    int i=1;
    Mat out;

    Ptr<SURF> detector_a = SURF::create(800);
    Ptr<SURF> detector_b = SURF::create(800);
    vector<Point2f> central_point;
    time_t start, end;
    
    Point2f p_zero = Point2f(0, 0);
    double sum_distance = 0;
    start = clock();
    while (i<101)
    {

    Mat img1 = imread(folder + img1_name + to_string(i) + extention, IMREAD_COLOR);
    Mat img2 = imread(folder + img2_name + to_string(i) + extention, IMREAD_COLOR);
    
    Point2f p = draw_match_double(img1, img2, out, detector_a, detector_b, min_size+100, min_size);
    
    sum_distance += get_distance(p, Point2f(608, 314));

    i++;
    }
    end = clock();
    double total_time = (end - start) * 1.0 / CLOCKS_PER_SEC;
    double distance_avg = sum_distance / 100;
    
    cout<<min_size<<","<<distance_avg<<","<<total_time<<endl;

}

int main(int argc, char const *argv[])
{

    for (int i = 100; i <= 300; i+= 25){

        func(i);
    }

    return 0;

}

