#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/***************************************************
 * 本节演示了如何根据前面训练的字典计算相似性评分
 * ************************************************/
int main( int argc, char** argv )
{
    // read the images and database  
    cout<<"reading database"<<endl;
    DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");  // use large vocab if you want: 
    if ( vocab.empty() )
    {
        cerr<<"Vocabulary does not exist."<<endl;
        return 1;
    }
    cout<<"reading images... "<<endl;
    vector<Mat> images; 
    for ( int i=0; i<10; i++ )
    {
        string path = "./data/"+to_string(i+1)+".png";
        images.push_back( imread(path) );
    }
    
    // NOTE: in this case we are comparing images with a vocabulary generated by themselves, this may leed to overfitting.  
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors;
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }
    
    // we can compare the images directly or we can compare one image to a database 
    // images :
    cout<<"comparing images with images "<<endl;
    for ( int i=0; i<images.size(); i++ )
    {
        DBoW3::BowVector v1;
        vocab.transform( descriptors[i], v1 );
        // Bow描述向量中含有每个单词的ID和权重
        if( i==0 )
          cout << v1 << endl;
        for ( int j=i; j<images.size(); j++ )
        {
            DBoW3::BowVector v2;
            vocab.transform( descriptors[j], v2 );
            double score = vocab.score(v1, v2);
            cout<<"image "<<i<<" vs image "<<j<<" : "<<score<<endl;
        }
        cout<<endl;
    }
    
    // or with database 
    cout<<"comparing images with database "<<endl;
    DBoW3::Database db( vocab, false, 0);
    for ( int i=0; i<descriptors.size(); i++ )
        db.add(descriptors[i]);
    cout<<"database info: "<<db<<endl;
    for ( int i=0; i<descriptors.size(); i++ )
    {
        DBoW3::QueryResults ret;
        db.query( descriptors[i], ret, 4);      // max result=4，显示四个匹配结果
        cout<<"searching for image "<<i<<" returns "<<ret<<endl<<endl;
    }
    cout<<"done."<<endl;
}
