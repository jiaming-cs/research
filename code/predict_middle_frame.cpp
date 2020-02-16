#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

double get_degree(Point2f p){
    if (p.x == 0)
        if(p.y > 0)
            return CV_PI / 2;
        else
            return CV_PI * 1.5;
    else{
        double degree = atan(p.y / p.x);
        if (p.x > 0){
            if (p.y > 0)
                return degree;
            else
                return 2 * CV_PI - degree;
        } 
        else{
            if (p.y > 0)
                return CV_PI - degree;
            else
                return CV_PI + degree;
        }
    }

}



Mat get_predict_descpritor_keypoints(Mat img1, Mat img2, Mat large, Mat bk, int k = 1, bool extended = true){
   
    int minHessian=400;
    Ptr<SURF>  detector = SURF::create(minHessian, 4, 3, extended);
    vector<KeyPoint>key_points_img1, key_points_img2, key_points_large;

    Mat descriptor_img1, descriptor_img2, descriptor_large;
    
    detector->detectAndCompute(img1, noArray(), key_points_img1, descriptor_img1);
    detector->detectAndCompute(img2, noArray(), key_points_img2, descriptor_img2);
    detector->detectAndCompute(large, noArray(), key_points_large, descriptor_large);
    

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
    
    double degree_sum = 0;
    double distance_sum = 0;
    vector<double> degrees;
    vector<double> distances;

    Point2f p1, p2;
    for (int i = 0; i < good_matches.size(); i++)
    {
        p1 = key_points_img1[good_matches[i].queryIdx].pt;
        p2 = key_points_img2[good_matches[i].trainIdx].pt;
        degrees.push_back(get_degree(p2 - p1));
        distances.push_back(norm(p2 - p1));
        distance_sum += norm(p2 - p1);
        degree_sum += get_degree(p2 - p1);
    }

    double distance_mean = distance_sum / good_matches.size();
    double degree_mean =  degree_sum / good_matches.size();
    
    double var_degree = 0;
    double var_distance = 0;
    for (int i = 0; i < good_matches.size(); i++){
        var_degree += (degrees[i] - degree_mean) * (degrees[i] - degree_mean); 
        var_distance += (distances[i] - distance_mean) * (distances[i] - distance_mean); 
    }

    double u_degree = sqrt(var_degree / (good_matches.size() - 1));
    double u_distance = sqrt(var_distance / (good_matches.size() - 1));

    vector<Point2f> predict_key_points;
    Mat predict_descriptor;

    for (int i = 0; i < good_matches.size(); i++)
    {
        p1 = key_points_img1[good_matches[i].queryIdx].pt;
        p2 = key_points_img2[good_matches[i].trainIdx].pt;
        double dis = norm(p2 - p1);
        double dg = get_degree(p2 - p1);
        if (dis < distance_mean - k * u_distance || dis > distance_mean + k * u_distance || dg < degree_mean - k * u_degree || dg > degree_mean + k * u_degree)
            continue;
        else{
            int query_index = good_matches[i].queryIdx;
            int train_index = good_matches[i].trainIdx;
            predict_key_points.push_back((key_points_img1[query_index].pt + key_points_img2[train_index].pt) / 2);
            Mat descriptor = (descriptor_img1.row(query_index) + descriptor_img2.row(train_index)) / 2;
            predict_descriptor.push_back(descriptor);
        }
    }


    matches.clear();
    matcher.knnMatch(predict_descriptor, descriptor_large, matches, 2);

    good_matches.clear();
    for(int i = 0; i < matches.size(); i++)
    {
        
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }

    }

    sort(good_matches.begin(), good_matches.end());

    vector<Point2f> large_points;
    vector<Point2f> prd_points;
 
    
    for(unsigned int i = 0; i < min(50, (int)good_matches.size()); ++i)
    {
        large_points.push_back(key_points_large[good_matches[i].trainIdx].pt);
        prd_points.push_back(predict_key_points[good_matches[i].queryIdx]);
    }

    Mat H = findHomography(prd_points, large_points, RANSAC);
    
    vector<Point2f> img1_corners(4);
    img1_corners[0]=Point2f(0,0);
    img1_corners[1]=Point2f(img1.cols,0);
    img1_corners[2]=Point2f(img1.cols, img1.rows);
    img1_corners[3]=Point2f(0,img1.rows);
    
    vector<Point2f> img2_corners(4);
    cout<<format( H, Formatter::FMT_PYTHON)<<endl;
    if (!H.empty()){
        
        perspectiveTransform(img1_corners, img2_corners, H);
        
        line(bk, img2_corners[0]+Point2f(img1.cols,0), img2_corners[1] + Point2f(img1.cols,0), Scalar(0, 0, 255), 2);
        line(bk, img2_corners[1]+Point2f(img1.cols,0), img2_corners[2] + Point2f(img1.cols,0), Scalar(0, 0, 255), 2);
        line(bk, img2_corners[2]+Point2f(img1.cols,0), img2_corners[3] + Point2f(img1.cols,0), Scalar(0, 0, 255), 2);
        line(bk, img2_corners[3]+Point2f(img1.cols,0), img2_corners[0] + Point2f(img1.cols,0), Scalar(0, 0, 255), 2);

    }
    return bk;
    
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
    string pre_img_name = "2fps/1.png";
    string middle_img_name = "2fps/2.png";
    string next_img_name = "2fps/3.png";
    string large_img_name = "large.png";

    Mat pre_img = imread(folder + pre_img_name, IMREAD_COLOR);
    Mat middle_img = imread(folder + middle_img_name, IMREAD_COLOR);
    Mat next_img = imread(folder + next_img_name, IMREAD_COLOR);
    Mat large_img = imread(folder + large_img_name, IMREAD_COLOR);

    Mat real_match = drawMatch(middle_img, large_img);
    Mat predict_match = get_predict_descpritor_keypoints(pre_img, next_img, large_img, real_match);
    imshow("Prediction", predict_match);
    waitKey(0);
    return 0;
}
