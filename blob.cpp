//
//  blob.cpp
//
//
//  Created by Jackson Beadle on 2/12/17.
//
//

#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include "exif.h"

#define PI 3.14159265

using namespace cv;
using namespace std;

easyexif::EXIFInfo info;
static double latitude;
static double longitude;
static double altitude;
static double PIXELUNIT;
static int width;
static int height;


struct Keycoord_t {
    double x;           /* pixels */
    double y;           /* pixels */
    double xmeters;     /* meters */
    double ymeters;     /* meters */
    double lat;         /* GPS latitude */
    double lon;         /* GPS longitude */
};

typedef vector<struct Keycoord_t *> Keys;
typedef struct Keycoord_t KeyCoordinates;

/* corners of images, clockwise starting in upper-left */
static KeyCoordinates corners[4];

void metersToGPS(KeyCoordinates* k) {
    double rad = 6378137;

    double midwidth = width/2;
    double midheight = height/2;

    /* difference in Lat/Lon (m -> GPS) */
    double diffLat = (k->ymeters - midheight)/rad;
    double diffLon = (k->xmeters - midwidth)/(rad*cos(PI*latitude/180));

    /* offset Lat/Lon */
    double offsetLat = latitude + diffLat * 180/PI;
    double offsetLon = longitude + diffLon * 180/PI;

    /* set Lat/Lon of KeyPoints */
    k->lat = offsetLat;
    k->lon = offsetLon;

}

void setupCorners() {

    corners[0].x = 0;
    corners[0].y = 0;
    corners[0].xmeters = 0;
    corners[0].ymeters = 0;
    metersToGPS(&corners[0]);

    corners[1].x = width;
    corners[1].y = 0;
    corners[1].xmeters = width*PIXELUNIT/1000;
    corners[1].ymeters = 0;
    metersToGPS(&corners[1]);

    corners[2].x = width;
    corners[2].y = height;
    corners[2].xmeters = width*PIXELUNIT/1000;
    corners[2].ymeters = height*PIXELUNIT/1000;
    metersToGPS(&corners[2]);

    corners[3].x = 0;
    corners[3].y = height;
    corners[3].xmeters = 0;
    corners[3].ymeters = height*PIXELUNIT/1000;
    metersToGPS(&corners[3]);

    for (int i = 0; i < 4; i++) {
        cout << "Corner: " << i+1 << endl;
        cout << "\t(X,Y): (" << corners[i].x << "," << corners[i].y << ")" << endl;
        cout << "\t(X,Y) (meters): (" << corners[i].xmeters << ","
             << corners[i].ymeters << ")" << endl;
        cout << "\tGPS: " << corners[i].lat << " latitude, "
             << corners[i].lon << " longitude" << endl;
    }

}


/*
 *  Function:    blob()
 *
 *  Description: Function reads grayscale binary color data from
 *               JPEG file and initalizes a cv::SimpleBlobDetector
 *               for file. Blob detector parameters set for multi-
 *               spectral images. For each centroid of a blob detected
 *               function creates new coordinates struct to convert
 *               from (X,Y) coordinates to GPS coordinates.
 *
 */
void blob(string src, Keys* keys) {


    /* my code */
    /* read in image from src path */
    Mat img = imread("input/"+src,IMREAD_GRAYSCALE);
    cout << "image opened" << endl;
    cout << "width: " << img.cols << endl;
    cout << "height: " << img.rows << endl;

    /* setup parameters */
    SimpleBlobDetector::Params params;

    /* thresholds */
    params.minThreshold = 10;
    params.maxThreshold = 200;
    params.minDistBetweenBlobs = 500;

    /*filter by color */
    params.filterByColor = false;
    params.blobColor = 145;

    /* filter by area */
    params.filterByArea = false;
    params.minArea = 1500;

    /* filter by circularity */
    params.filterByCircularity = false;
    params.minCircularity = 0.1;

    /* filter by convexity */
    params.filterByConvexity = false;
    params.minConvexity = 0.87;

    /* filter by inertia */
    params.filterByInertia = false;
    params.minInertiaRatio = 0.01;

    cout << "parameters setup" << endl;

    /* setup detector */
    SimpleBlobDetector detector(params);

    cout << "blob detector initialized" << endl;

    /* detect blobs */
    vector<KeyPoint> keypoints;
    detector.detect(img,keypoints);

    cout << "keypoints detected" << endl;

    /* show blobs as red */
    Mat img_keypoints;
    drawKeypoints(img, keypoints, img_keypoints, Scalar(0,0,225), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cout << "keypoints drawn" << endl;

    /* write image to file */
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
    try {
        imwrite("output/"+src,img_keypoints,compression_params);
        cout << "output image written" << endl;
    } catch (runtime_error& ex) {
        cerr << "Exception writing blob image";
    }

    /* writing points */
    vector<KeyPoint>::iterator it;
    KeyCoordinates *temp;
    for (it = keypoints.begin(); it != keypoints.end(); ++it) {
        //cout << "x: " << it->pt.x << " y: " << it->pt.y << endl;
        //cout << "\t(in m) x: " << (it->pt.x*PIXELUNIT)/1000 << " y: " << (it->pt.y*46.3)/1000 << endl;
        temp = new KeyCoordinates;
        temp->x = it->pt.x;
        temp->y = it->pt.y;
        temp->xmeters = it->pt.x*PIXELUNIT/1000;
        temp->ymeters = it->pt.y*PIXELUNIT/1000;
        metersToGPS(temp);
        keys->push_back(temp);
    }

    return;
}

/*  Function:    readexif()
 *
 *  Description: reads EXIF data from src JPEG file. Sets file GPS
 *               coordinates to global latitude and longitude for
 *               waypoint reference.
 *
 */
void readexif(string src) {

    string srcpath = "input/" + src;
	FILE *fp = fopen(srcpath.c_str(),"rb");
    if (!fp) {
        cout << "Can't open file for exif" << endl;
        return;
    }
    fseek(fp,0,SEEK_END);
    unsigned long fsize = ftell(fp);
    rewind(fp);
    unsigned char* buf = new unsigned char[fsize];
    if (fread(buf,1,fsize,fp) != fsize) {
        cout << "Can't read file for exif" << endl;
        delete[] buf;
        return;
    }
    fclose(fp);

    int code = info.parseFrom(buf,fsize);
    delete[] buf;
    if (code) {
        cout << "Error parsing EXIF: code " << code << endl;
        return;
    }

    cout << "GPS Latitude: " << info.GeoLocation.Latitude << endl;
    cout << "GPS Longitude: " << info.GeoLocation.Longitude << endl;

    /* set global GPS attributes for FILE */
    latitude = info.GeoLocation.Latitude;
    longitude = info.GeoLocation.Longitude;
    altitude = info.GeoLocation.Altitude;
    width = info.ImageWidth;
    height = info.ImageHeight;
}

void log_keys(Keys* keys) {
    KeyCoordinates* temp;
    for (unsigned i = 0; i < keys->size(); i++) {
        temp = keys->at(i);
        cout << "Keypoint: " << i << endl;
        cout << "\t(X,Y): (" << temp->x << "," << temp->y << ")" << endl;
        cout << "\t(X,Y) (meters): (" << temp->xmeters << ","
             << temp->ymeters << ")" << endl;
        cout << "\tGPS: " << temp->lat << " latitude, "
             << temp->lon << " longitude" << endl;
    }
}

int main(int argc, char* argv[]) {

    string src = argv[1];
    PIXELUNIT = atof(argv[2]);
    Keys* keys = new Keys;

    readexif(src);
    setupCorners();
    blob(src,keys);
    log_keys(keys);


    exit(EXIT_SUCCESS);
}
