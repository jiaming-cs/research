// test frame 83 84

#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <map>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include <iostream>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

Mat read_h(string file_name){
    Mat out = Mat::zeros(2, 3, CV_64FC1);
    ifstream in(file_name);
    double n;
    for (int i = 0; i<2 ; i++){
        for (int j = 0; j < 3; j++){
            in >> n;
            out.at<double>(i, j) = n;
        }
    }
    //cout<<format(out, Formatter::FMT_PYTHON)<<endl;
    return out;

}

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

double get_mse(const vector<Point2f> &img1_points, const vector<Point2f> &img2_points, Mat h, Mat h_gt){
    double error=0;
    Point2f p, p_gt;

    for(int i = 0 ; i < img1_points.size(); i++){
        p = get_position(img1_points[i], h);
        p_gt = get_position(img1_points[i], h_gt);
        error += get_distance(p, p_gt);
        
    }
    
    return error / img1_points.size();
        
     
}

void draw_match_double(Mat& img1,
                Mat& img2,
                Mat& img_matches,
                Ptr<SURF> &detector1,
                Ptr<SURF> &detector2,
                string index_1,
                string index_2,
                int max_size = 200,
                int min_size = 100,
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

    Mat H1 = findHomography(img1_points, img2_points, RHO, 3);
    Mat H2 = findHomography(img1_points, img2_points, RHO, 8);
    //Mat H3 = findHomography(img1_points, img2_points, RHO, 6);
    Mat H = get_better_h(img1_points, img2_points, H1, H2);
    //H = get_better_h(img1_points, img2_points, H, H3);
    
    Mat h = read_h("/home/jiaming/Documents/research/dataset/wall/H" + index_1+"to"+index_2+"p");
    cout<< get_mse(img1_points, img2_points, H, h)<<endl;
    
    
}

void draw_match_single(Mat& img1,
                Mat& img2,
                Mat& img_matches,
                Ptr<SURF> &detector1,
                Ptr<SURF> &detector2,
                string index_1,
                string index_2,
                int max_size = 200,
                int min_size = 100,
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

    Mat H = findHomography(img1_points, img2_points, RANSAC, 3);
    Mat h = read_h("/home/jiaming/Documents/research/dataset/wall/H" + index_1 +"to"+index_2+"p");
    cout<< get_mse(img1_points, img2_points, H, h)<<endl;
   
    
  
}


void func(string dataset_name, string index_1, string index_2){

    string folder = "/home/jiaming/Documents/research/dataset/";
    string extention = ".ppm";
    Mat out;

    Ptr<SURF> detector_a = SURF::create(1000);
    Ptr<SURF> detector_b = SURF::create(1000);
    vector<Point2f> central_point;
    time_t start, end;
    Mat img1 = imread(folder + dataset_name + "/img" + index_1 + extention, IMREAD_COLOR);
    Mat img2 = imread(folder + dataset_name + "/img" + index_2 + extention, IMREAD_COLOR);
    
    
    draw_match_single(img1, img2, out, detector_a, detector_b, index_1, index_2);

    detector_a = SURF::create(1000);
    detector_b = SURF::create(1000);
   
    draw_match_double(img1, img2, out, detector_a, detector_b, index_1, index_2);
   
}

int main(int argc, char const *argv[])
{

    if (argc != 4){
        cout<< "<dataset name> <index_1> <index_2>" <<endl;
        exit(1);
    }
    string dataset_name = argv[1];
    string index_1 = argv[2];
    string index_2 = argv[3];

    func(dataset_name, index_1, index_2);
    return 0;

}

