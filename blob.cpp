//
//  blob.cpp
//
//
//  Created by Jackson Beadle on 2/12/17.
//  Last modified by Jackson Beadle on 6/9/17.
//
//  This file contains all the implementation of the feature
//  recognition software to be included in the Rail Launch UAV
//  senior design project, an interdisciplinary project between
//  the mechanical and computer engineering departments at
//  Santa Clara University.
//
//  Program takes input (file name of image) from an "input"
//  directory located with this file. Output image, if written
//  is stored in an "output" directory with the same file name
//  as the input file. Standard out prints wherever directed.
//
//  This program utilizes the OpenCV computer vision library
//  for its blob detection functionality.
//
//  Metadata from images is read using the EasyEXIF library, sourced
//  from its GitHub repository at:
//  https://github.com/mayanklahiri/easyexif
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


/* define pi with more sig figs */
#define PI 3.14159265


using namespace cv;
using namespace std;


/* global variables */
easyexif::EXIFInfo info;
static double latitude;
static double longitude;
static double altitude;
static double PIXELUNIT;
static int width;
static int height;


/* struct:    Keycoord_t
 *
 * Description: data struct to hold pixel values, meter offsets,
 *              and GPS coordinates in two-dimensions. This is the
 *              main data type for calculating GPS coordinates for
 *              each blob (i.e. vegetation feature)
 *
 */

struct Keycoord_t {
    double x;           /* pixels */
    double y;           /* pixels */
    double xmeters;     /* meters */
    double ymeters;     /* meters */
    double lat;         /* GPS latitude */
    double lon;         /* GPS longitude */
    Keycoord_t() { x = y = xmeters = ymeters = lat = lon = 0; }
    Keycoord_t(double latitude, double longitude) {
        lat = latitude;
        lon = longitude;
    }
};

/* typedefs */
typedef vector<struct Keycoord_t *> Keys;
typedef struct Keycoord_t KeyCoordinates;

/* corners of images, clockwise starting in upper-left */
static KeyCoordinates corners[4];



/* Function:    metersToGPS(KeyCoordinates* k)
 *
 * Description: Function takes a KeyCoordinates struct with pixels
 *              value and meter offsets. Using this data and the
 *              GPS metadata from the image, can convert to GPS
 *              coordinates.
 *
 */

void metersToGPS(KeyCoordinates* k) {
    double rad = 6378137;

    /* difference in Lat/Lon (m -> GPS) */
    double diffLat = k->ymeters/rad;
    double diffLon = k->xmeters/(rad*cos(PI*latitude/180));

    /* offset Lat/Lon */
    double offsetLat = latitude + diffLat * 180/PI;
    double offsetLon = longitude + diffLon * 180/PI;

    /* set Lat/Lon of KeyPoints */
    k->lat = offsetLat;
    k->lon = offsetLon;

}


/* FunctionL     setupCorners()
 *
 * Description: Function establishes the four corners of the image,
 *              in pixel and GPS coordiantes.
 *
 */

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


/* Function:    boundaries(Keys* keys)
 *
 * Description: Function finds minimum and maximum latitude
 *              and longitude to establish boundaries of area
 *              of interest. Stores boundaries in Keys vector
 *              to be logged later.
 */

Keys *boundaries(Keys* keys) {
    Keys* b = new Keys;
    KeyCoordinates *nw, *ne, *se, *sw;

    double minLat, maxLat, minLon, maxLon;
    KeyCoordinates *temp = keys->at(0);
    minLat = maxLat = temp->lat;
    minLon = maxLon = temp->lon;

    /* find min and max latitude and longitude */
    for (unsigned i = 1; i < keys->size(); i++) {
        temp = keys->at(i);
        if (temp->lat < minLat)
            minLat = temp->lat;
        else if (temp->lat > maxLat)
            maxLat = temp->lat;
        if (temp->lon < minLon)
            minLon = temp->lon;
        else if (temp->lon > maxLon)
            maxLon = temp->lon;
    }

    nw = new KeyCoordinates(minLat, minLon);
    b->push_back(nw);
    ne = new KeyCoordinates(minLat, maxLon);
    b->push_back(ne);
    se = new KeyCoordinates(maxLat, maxLon);
    b->push_back(se);
    sw = new KeyCoordinates(maxLat, minLon);
    b->push_back(sw);

    return b;
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


    /* read in image from src path */
    Mat img_rgb = imread("input/"+src/*,IMREAD_GRAYSCALE*/);
    Mat img;
    inRange(img_rgb, Scalar(43,17,36), Scalar(101,65,79), img);
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
    params.filterByColor = true;
    params.blobColor = 255;

    /* filter by area */
    params.filterByArea = true;
    params.minArea = 200;

    /* filter by circularity */
    params.filterByCircularity = false;
    params.minCircularity = 0.9;

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
    // Mat img_keypoints;
    // drawKeypoints(img, keypoints, img_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cout << "keypoints drawn" << endl;

    /* write image to file */
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
    try {
        // imwrite for outputing image with keypoints drawn
        // imwrite("output/"+src,img_keypoints,compression_params);

        // imwrite for outputing black/white image
        // imwrite("output/"+src,img,compression_params);
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
        if (isnan(it->pt.x) || isnan(it->pt.y))
            continue;
        temp->x = it->pt.x;
        temp->y = it->pt.y;
        temp->xmeters = (it->pt.x-width/2)*PIXELUNIT/1000;
        temp->ymeters = (it->pt.y-height/2)*PIXELUNIT/1000;
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


/* Function:    log_keys(Keys* keys)
 *
 * Description: This function takes in a vector of KeyCoordinates,
 *              i.e. the vector of keypoints detected by the blob
 *              analysis. All data stored in each keypoint is printed
 *              to standard output for logging.
 *
 * Recommend:  route all stdout to a log file (i.e. ./blob > out.log)
 *
 */

void log_keys(Keys* keys) {

    Keys* bound = boundaries(keys);
    KeyCoordinates *nw, *ne, *se, *sw;

    nw = bound->at(0);
    ne = bound->at(1);
    se = bound->at(2);
    sw = bound->at(3);

    cout << endl << "Boundaries of interest area:" << endl;
    cout << "\t" << nw->lat << "\t\t\t" << ne->lat << endl;
    cout << "\t" << nw->lon << "\t\t\t" << ne->lon << endl;
    cout << endl << endl;
    cout << "\t" << sw->lat << "\t\t\t" << se->lat << endl;
    cout << "\t" << sw->lon << "\t\t\t" << se->lon << endl;

    cout << endl << "Centroids of blobs:" << endl;
    KeyCoordinates* temp;
    for (unsigned i = 0; i < keys->size(); i++) {
        temp = keys->at(i);
        cout << "Keypoint: " << i << endl;
        cout << "\t(X,Y): (" << temp->x << "," << temp->y << ")" << endl;
        cout << "\t(X,Y) offset (meters): (" << temp->xmeters << ","
             << temp->ymeters << ")" << endl;
        cout << "\tGPS: " << temp->lat << " latitude, "
             << temp->lon << " longitude" << endl;
    }
}


/* Function:    main(int argc, char* argvp[])
 *
 * Description: Main function to read metadata, conduct blob analysis,
 *              and log all results to standard output.
 *
 * Execution:   "./blob <img_file> <pixelunit> > <log_file>"
 *
 */

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cout << "Missing Arguments: ./blob <img> <pixelunit>" << endl;
        return -1;
    }

    string src = argv[1];
    PIXELUNIT = atof(argv[2]);
    Keys* keys = new Keys;

    readexif(src);
    setupCorners();
    blob(src,keys);
    log_keys(keys);


    exit(EXIT_SUCCESS);
}
