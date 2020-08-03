#define _WIN32_WINNT 0x0501
#include <windows.h>
#include <iostream>

using namespace std;
#pragma comment(lib,"shell32.lib")


#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include "cv.h"
#include "highgui.h"

#endif
# include"math.h"
#include <stdio.h>
#include <ctype.h>
#include < cvtypes.h>
#include <cxerror.h>
#include <cxcore.h>
#include <cvver.h>
#include <cvcompat.h>
#include <basetsd.h>




#include<stdlib.h>


int mb1=0,mb2=0,moveflag=0,paintflg=0; 

IplImage *image = 0, *hsv = 0, *hue = 0, *mask = 0, *backproject = 0, *histimg = 0;
CvHistogram *hist = 0;

int backproject_mode = 0,keyboardflg=0,keyboardflg1=2;
int select_object = 0;
int track_object = 0;
int show_hist = 1;
int f=0,cnt=0,cnt1=0,cnt2=0;
int hdims = 16;
float hranges_arr[] = {0,180};
float* hranges = hranges_arr;
int vmin = 10, vmax = 256, smin = 30;
int x=0,y=0,x2=0,y2=0;
float tempx=0.0;
float tempy=0.0;
void MoveMouse(int x,int y);

CvPoint origin;
CvRect selection;
CvRect track_window;
CvBox2D track_box;
CvConnectedComp track_comp;

#define SAMPLING 7
#include <math.h>
#define M_PI 3.14
#define ZERO 0.000000

void initData(int);
void initWeights(void);
void forward_pass();
void backward_pass();
void learn();
void calcNet(void);
void calcOverallError(void);
void sendchar(int);
void gesture();

IplImage* color_img0;
IplImage* color_img;
IplImage* gray_img0 = NULL;
IplImage* gray_img = NULL;
IplImage* dilate_img = NULL;
IplImage* erosion_img = NULL;

IplConvKernel* element = 0;
int element_shape = CV_SHAPE_RECT;

IplImage* thresh = 0;
// Edge detection
IplImage * edgeImage; //Declare an image for holding the results of the edge detection perfomed on the grayScaleImage.

// A Simple Camera Capture Framework


int edge_thresh = 100,fl=0;
int keyflag=0;

int numPatterns=2,numInputs=7,numHidden=7,numOutput=2;
const double learning_rate_IH=0.7;
const double learning_rate_HO=0.07;
const int numEpochs = 500;

double weightsIH[7][7];
double weightsHO[7][2];
double result;
double candidate[5000],input[7],hidden[7],output[2],errorsignal_hidden[7],errorsignal_output[7],target[2][2]={0.538146,0,0,0};

//// variables ////
int patNum = 0,pattern;
double errThisPat = 0.0;
double outPred = 0.0;
double RMSerror = 0.0;
//double trainInputs[numPatterns][numInputs];
FILE *fp,*fp1;

struct storage
{
   double temp_ip_to_hidden[7];
   
}s[7];


struct storage1
{
    double temp_hidden_to_op[7];
}s1[7];

////////////////////////////////////////////////

void on_mouse( int event, int x, int y, int flags, void* param )
{
    if( !image )
        return;

    if( image->origin )
        y = image->height - y;

    if( select_object )
    {
        selection.x = MIN(x,origin.x);
        selection.y = MIN(y,origin.y);
        selection.width = selection.x + CV_IABS(x - origin.x);
        selection.height = selection.y + CV_IABS(y - origin.y);
        
        selection.x = MAX( selection.x, 0 );
        selection.y = MAX( selection.y, 0 );
        selection.width = MIN( selection.width, image->width );
        selection.height = MIN( selection.height, image->height );
        selection.width -= selection.x;
        selection.height -= selection.y;
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = cvPoint(x,y);
        selection = cvRect(x,y,0,0);
        select_object = 1;
        break;
    case CV_EVENT_LBUTTONUP:
        select_object = 0;
        if( selection.width > 0 && selection.height > 0 )
            track_object = -1;
        break;
    }
}


CvScalar hsv2rgb( float hue )
{
    int rgb[3], p, sector;
    static const int sector_data[][3]=
        {{0,2,1}, {1,2,0}, {1,0,2}, {2,0,1}, {2,1,0}, {0,1,2}};
    hue *= 0.033333333333333333333333333333333f;
    sector = cvFloor(hue);
    p = cvRound(255*(hue - sector));
    p ^= sector & 1 ? 255 : 0;

    rgb[sector_data[sector][0]] = 255;
    rgb[sector_data[sector][1]] = 0;
    rgb[sector_data[sector][2]] = p;

    return cvScalar(rgb[2], rgb[1], rgb[0],0);
}

int main( int argc, char** argv )
{
    CvCapture* capture = 0;
    int gta1=0;
    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        capture = cvCaptureFromCAM( argc == 2 ? argv[1][0] - '0' : 0 );
    else if( argc == 2 )
        capture = cvCaptureFromAVI( argv[1] ); 

    if( !capture )
    {
        fprintf(stderr,"Could not initialize capturing...\n");
        return -1;
    }

    printf( "Hot keys: \n"
        "\tESC - quit the program\n"
        "\tc - stop the tracking\n"
        "\tb - switch to/from backprojection view\n"
        "\th - show/hide object histogram\n"
        "To initialize tracking, select the object with mouse\n" );

    cvNamedWindow( "Histogram", 1 );
    cvNamedWindow( "CamShiftDemo", 1 );
    cvSetMouseCallback( "CamShiftDemo", on_mouse, 0 );
    cvCreateTrackbar( "Vmin", "CamShiftDemo", &vmin, 256, 0 );
    cvCreateTrackbar( "Vmax", "CamShiftDemo", &vmax, 256, 0 );
    cvCreateTrackbar( "Smin", "CamShiftDemo", &smin, 256, 0 );

///////////////////////////////////////
	IplImage* frame = 0;
        int i, bin_w, c;
    for(;;)
    {
        //IplImage* frame = 0;
        //int i, bin_w, c;

        frame = cvQueryFrame( capture );
        if( !frame )
            break;

        if( !image )
        {
            /* allocate all the buffers */
            image = cvCreateImage( cvGetSize(frame), 8, 3 );
            image->origin = frame->origin;
            hsv = cvCreateImage( cvGetSize(frame), 8, 3 );
            hue = cvCreateImage( cvGetSize(frame), 8, 1 );
            mask = cvCreateImage( cvGetSize(frame), 8, 1 );
            backproject = cvCreateImage( cvGetSize(frame), 8, 1 );
            hist = cvCreateHist( 1, &hdims, CV_HIST_ARRAY, &hranges, 1 );
            histimg = cvCreateImage( cvSize(320,200), 8, 3 );
            cvZero( histimg );
        }

        cvCopy( frame, image, 0 );
        cvCvtColor( image, hsv, CV_BGR2HSV );

        if( track_object )
        {
            int _vmin = vmin, _vmax = vmax;

            cvInRangeS( hsv, cvScalar(0,smin,MIN(_vmin,_vmax),0),
                        cvScalar(180,256,MAX(_vmin,_vmax),0), mask );
            cvSplit( hsv, hue, 0, 0, 0 );

            if( track_object < 0 )
            {
                float max_val = 0.f;
                cvSetImageROI( hue, selection );
                cvSetImageROI( mask, selection );
                cvCalcHist( &hue, hist, 0, mask );
                cvGetMinMaxHistValue( hist, 0, &max_val, 0, 0 );
                cvConvertScale( hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0 );
                cvResetImageROI( hue );
                cvResetImageROI( mask );
                track_window = selection;
                track_object = 1;

                cvZero( histimg );
                bin_w = histimg->width / hdims;
                for( i = 0; i < hdims; i++ )
                {
                    int val = cvRound( cvGetReal1D(hist->bins,i)*histimg->height/255 );
                    CvScalar color = hsv2rgb(i*180.f/hdims);
                    cvRectangle( histimg, cvPoint(i*bin_w,histimg->height),
                                 cvPoint((i+1)*bin_w,histimg->height - val),
                                 color, -1, 8, 0 );
					////
				//	x=track_window.x;
				//	y=track_window.y;
				//	printf("\n%d\t%d",x,y);
					
                }
            }

            cvCalcBackProject( &hue, backproject, hist );
            cvAnd( backproject, mask, backproject, 0 );
            cvCamShift( backproject, track_window,
                        cvTermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ),
                        &track_comp, &track_box );
            track_window = track_comp.rect;
            if( backproject_mode )
                cvCvtColor( backproject, image, CV_GRAY2BGR );
            if( image->origin )
                track_box.angle = -track_box.angle;
			
		//	printf("%f\n",track_box.angle);
            
				cvEllipseBox( image, track_box, CV_RGB(255,0,0), 3, CV_AA, 0 );
			////
					x=track_window.x;
					y=480-track_window.y;
					tempx=x*1.8;
					x=floor(tempx);
					tempy=y*1.8;
					y=floor(tempy);
				//	printf("%d\t%d\n",x,y);
					//abs(x-x2)>1 && abs(y-y2)>2				
					if((abs(x-x2)>1 && abs(y-y2)>2)&&(track_box.angle>65.0 && track_box.angle<90.0)||(track_box.angle<-77.0 && track_box.angle>-89.0))
					{
						cnt2++;
						 if (f==0) //&& cnt2<5)
							MoveMouse(x,y);

					}
			//		else
			//			cnt2=0;

					
					
					x2=x;
					y2=y;
				//	printf("\n%d\t%d",x,y);
                   if(GetAsyncKeyState(71))	
				   {
					   f=1;
					   break;
				   }
				  
					//CODE FOR DOUBLE CLICK
				if(track_box.angle>-50.0 && track_box.angle<-1.0)
					cnt1++;
				else
					cnt1=0;
				gta1=0;
				if(cnt1>=2)
				{
					cnt1=0;
					 mouse_event(MOUSEEVENTF_LEFTDOWN, x, y, 0, 0); //Click Down
                     mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Up 
					 
					 mouse_event(MOUSEEVENTF_LEFTDOWN, x, y, 0, 0); //Click Down
                     mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Up 
					 Sleep(500);
					 HWND windowHandle3 = FindWindow(NULL,"GTA: Vice City");
					 if(windowHandle3 != NULL)
					 {
						  gta1=1;
						  Sleep(2500);
					 }

				}	
				//CODE FOR SINGLE CLICK
				if(track_box.angle<55.0 && track_box.angle>10.0)
					cnt++;
				else
					cnt=0;
				if(cnt>=2 || gta1==1)
				{
				
					cnt=0;
					if(gta1==0)
					{
					 mouse_event(MOUSEEVENTF_LEFTDOWN, x, y, 0, 0); //Click Down
                     mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Up 
					 Sleep(200);
					}
						gta1=0;
					 //FOR TYPING IN NOTEPAD OR WORDPAD
					// HWND windowHandle4 = FindWindow(NULL,"Document - WordPad");

					 HWND windowHandle = FindWindow(NULL,"Untitled - Notepad");
					 if(windowHandle != NULL)// || windowHandle4 !=NULL)
					 {
						 cvReleaseCapture( &capture );
					//	 Sleep(2000);
						 gesture();


						 //FOR RETURNING TO MOUSE CONTROL
						 if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
							capture = cvCaptureFromCAM( argc == 2 ? argv[1][0] - '0' : 0 );
						else if( argc == 2 )
							capture = cvCaptureFromAVI( argv[1] ); 

						if( !capture )
						{
							fprintf(stderr,"Could not initialize capturing...\n");
							return -1;
						}

						printf( "Hot keys: \n"
						"\tESC - quit the program\n"
						"\tc - stop the tracking\n"
						"\tb - switch to/from backprojection view\n"
						"\th - show/hide object histogram\n"
						"To initialize tracking, select the object with mouse\n" );

						cvNamedWindow( "Histogram", 1 );
						cvNamedWindow( "CamShiftDemo", 1 );
						cvSetMouseCallback( "CamShiftDemo", on_mouse, 0 );
						cvCreateTrackbar( "Vmin", "CamShiftDemo", &vmin, 256, 0 );
						cvCreateTrackbar( "Vmax", "CamShiftDemo", &vmax, 256, 0 );
						cvCreateTrackbar( "Smin", "CamShiftDemo", &smin, 256, 0 );

					 }

					 //	FOR GAME CONTROL GTA VICE CITY 
					
					  HWND windowHandle3 = FindWindow(NULL,"GTA: Vice City");

					  if(windowHandle3 != NULL)
					  {
						  HWND windowHandle11 = FindWindow(NULL,"GTA: Vice City");

						  while(track_box.angle!=0.0 && windowHandle11 != NULL)
						  {
						//	printf("\n %f",track_box.angle);
			//////////		   		if(track_box.angle<65.0 && track_box.angle>3.0 && mb1==0)
							  if(track_box.angle<65.0 && track_box.angle>3.0)
							{
						
								keyboardflg=0;
								
								mouse_event(MOUSEEVENTF_LEFTDOWN, x, y, 0, 0); //Click LEFT Down
							
								mouse_event(MOUSEEVENTF_MIDDLEDOWN, x, y, 0, 0); //Click MIDDLE Down
								mb1=1;



						
							}	
							//if(track_box.angle>-65.0 && track_box.angle<-1.0 && mb2==0)
							  if(track_box.angle>-65.0 && track_box.angle<-1.0)
							{
								keyboardflg=0;		
								
								mouse_event(MOUSEEVENTF_RIGHTDOWN, x, y, 0, 0); //Click Down
						
								mouse_event(MOUSEEVENTF_MIDDLEDOWN, x, y, 0, 0); //Click MIDDLE Down
								mb2=1;
							}
							//printf("%d\n",track_window);
							if((track_box.angle>65.0 && track_box.angle<90.0)||(track_box.angle<-77.0 && track_box.angle>-89.0))
							{
				/////////				if(keyboardflg==1)
				//////////					keyboardflg1=1;
				///////////				if(keyboardflg==0)
				//////////				{
				///////////					keyboardflg=1;
				/////////					keyboardflg1=0;
				//////////				}
								//FOR MOVING IN FORWARD DIRECTION WHEN X > 240
								    mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Down
									mouse_event(MOUSEEVENTF_RIGHTUP, x, y, 0, 0); //Click Down
								
							}
							if(track_window.y>240)
									mouse_event(MOUSEEVENTF_MIDDLEDOWN, x, y, 0, 0); //Click MIDDLE Down
								else //if(track_window.x<240)
								{
									mouse_event(MOUSEEVENTF_MIDDLEDOWN, x, y, 0, 0); //Click MIDDLE Down
									mouse_event(MOUSEEVENTF_MIDDLEUP, x, y, 0, 0); //Click MIDDLE UP
								}
							/*if(keyboardflg==1 && keyboardflg1==0)
							{
								if(mb1==1)
								{
									mb1=0;
								mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Down
								}
								if(mb2==1)
								{
									mb2=0;
								mouse_event(MOUSEEVENTF_RIGHTUP, x, y, 0, 0); //Click Down
								}
								
								
						
							}*/
					

							frame = cvQueryFrame( capture );
							if( !frame )
								break;

							if( !image )
							{
									/* allocate all the buffers */
								image = cvCreateImage( cvGetSize(frame), 8, 3 );
								image->origin = frame->origin;
								hsv = cvCreateImage( cvGetSize(frame), 8, 3 );
								hue = cvCreateImage( cvGetSize(frame), 8, 1 );
								mask = cvCreateImage( cvGetSize(frame), 8, 1 );
								backproject = cvCreateImage( cvGetSize(frame), 8, 1 );
								hist = cvCreateHist( 1, &hdims, CV_HIST_ARRAY, &hranges, 1 );
								histimg = cvCreateImage( cvSize(320,200), 8, 3 );
								cvZero( histimg );
							}

							cvCopy( frame, image, 0 );
							cvCvtColor( image, hsv, CV_BGR2HSV );

							if( track_object )
							{
								int _vmin = vmin, _vmax = vmax;

								cvInRangeS( hsv, cvScalar(0,smin,MIN(_vmin,_vmax),0),
												cvScalar(180,256,MAX(_vmin,_vmax),0), mask );
								cvSplit( hsv, hue, 0, 0, 0 );

								if( track_object < 0 )
								{
									float max_val = 0.f;
									cvSetImageROI( hue, selection );
									cvSetImageROI( mask, selection );
									cvCalcHist( &hue, hist, 0, mask );
									cvGetMinMaxHistValue( hist, 0, &max_val, 0, 0 );
									cvConvertScale( hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0 );
									cvResetImageROI( hue );
									cvResetImageROI( mask );
									track_window = selection;
									track_object = 1;

									cvZero( histimg );
									bin_w = histimg->width / hdims;
									for( i = 0; i < hdims; i++ )
									{
										int val = cvRound( cvGetReal1D(hist->bins,i)*histimg->height/255 );
										CvScalar color = hsv2rgb(i*180.f/hdims);
										cvRectangle( histimg, cvPoint(i*bin_w,histimg->height),
										cvPoint((i+1)*bin_w,histimg->height - val),
										color, -1, 8, 0 );
								
											
									}
								}
								cvCalcBackProject( &hue, backproject, hist );
								cvAnd( backproject, mask, backproject, 0 );
								cvCamShift( backproject, track_window,
								cvTermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ),
								&track_comp, &track_box );
								track_window = track_comp.rect;
								if( backproject_mode )
									cvCvtColor( backproject, image, CV_GRAY2BGR );
								if( image->origin )
									track_box.angle = -track_box.angle;
								
							//		printf("%f\n",track_box.angle);
							}
							windowHandle11 = FindWindow(NULL,"GTA: Vice City");			

						}
						mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Down
						mouse_event(MOUSEEVENTF_RIGHTUP, x, y, 0, 0); //Click Down
						mouse_event(MOUSEEVENTF_MIDDLEUP, x, y, 0, 0); //Click MIDDLE UP

					  }

					  HWND windowHandle2 = FindWindow(NULL,"untitled - Paint");
					  f=0;
					  if(windowHandle2 != NULL)
					  {
						  //HWND windowHandle = FindWindow(NULL,"Hand Gesture Recognition");
							SetWindowText(windowHandle2, "Paint");
						  while(1)
						  {
							
							if(GetAsyncKeyState(71))							
								break;
						//	printf("\n %f",track_box.angle);
							

					   		if(track_box.angle<65.0 && track_box.angle>3.0 && mb1==0)
							{
						
								keyboardflg=0;
						
								mouse_event(MOUSEEVENTF_LEFTDOWN, x, y, 0, 0); //Click Down
								mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Down
								mb1=1;



						
							}	
							if(track_box.angle>-65.0 && track_box.angle<-1.0 && mb2==0)
							{
								keyboardflg=0;		
								mouse_event(MOUSEEVENTF_LEFTDOWN, x, y, 0, 0); //Click Down
								mb2=1;
							}
							if((track_box.angle>65.0 && track_box.angle<90.0)||(track_box.angle<-77.0 && track_box.angle>-89.0))
							{
									mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Down
									mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0);
								if(keyboardflg==1)
									keyboardflg1=1;
								if(keyboardflg==0)
								{
									keyboardflg=1;
									keyboardflg1=0;
								}
							}
							if(keyboardflg==1 && keyboardflg1==0)
							{
								if(mb1==1)
								{
									mb1=0;
								mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Down
								}
								if(mb2==1)
								{
									mb2=0;
									mouse_event(MOUSEEVENTF_LEFTUP, x, y, 0, 0); //Click Down
								}
								
						
							}
					

							frame = cvQueryFrame( capture );
							if( !frame )
								break;

							if( !image )
							{
									/* allocate all the buffers */
								image = cvCreateImage( cvGetSize(frame), 8, 3 );
								image->origin = frame->origin;
								hsv = cvCreateImage( cvGetSize(frame), 8, 3 );
								hue = cvCreateImage( cvGetSize(frame), 8, 1 );
								mask = cvCreateImage( cvGetSize(frame), 8, 1 );
								backproject = cvCreateImage( cvGetSize(frame), 8, 1 );
								hist = cvCreateHist( 1, &hdims, CV_HIST_ARRAY, &hranges, 1 );
								histimg = cvCreateImage( cvSize(320,200), 8, 3 );
								cvZero( histimg );
							}

							cvCopy( frame, image, 0 );
							cvCvtColor( image, hsv, CV_BGR2HSV );

							if( track_object )
							{
								int _vmin = vmin, _vmax = vmax;

								cvInRangeS( hsv, cvScalar(0,smin,MIN(_vmin,_vmax),0),
												cvScalar(180,256,MAX(_vmin,_vmax),0), mask );
								cvSplit( hsv, hue, 0, 0, 0 );

								if( track_object < 0 )
								{
									float max_val = 0.f;
									cvSetImageROI( hue, selection );
									cvSetImageROI( mask, selection );
									cvCalcHist( &hue, hist, 0, mask );
									cvGetMinMaxHistValue( hist, 0, &max_val, 0, 0 );
									cvConvertScale( hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0 );
									cvResetImageROI( hue );
									cvResetImageROI( mask );
									track_window = selection;
									track_object = 1;

									cvZero( histimg );
									bin_w = histimg->width / hdims;
									for( i = 0; i < hdims; i++ )
									{
										int val = cvRound( cvGetReal1D(hist->bins,i)*histimg->height/255 );
										CvScalar color = hsv2rgb(i*180.f/hdims);
										cvRectangle( histimg, cvPoint(i*bin_w,histimg->height),
										cvPoint((i+1)*bin_w,histimg->height - val),
										color, -1, 8, 0 );
								
											
									}
								}
								cvCalcBackProject( &hue, backproject, hist );
								cvAnd( backproject, mask, backproject, 0 );
								cvCamShift( backproject, track_window,
								cvTermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ),
								&track_comp, &track_box );
								track_window = track_comp.rect;
								if( backproject_mode )
									cvCvtColor( backproject, image, CV_GRAY2BGR );
								if( image->origin )
									track_box.angle = -track_box.angle;
								x=track_window.x;
								y=480-track_window.y;
								tempx=x*1.8;
								x=floor(tempx);
								tempy=y*1.8;
								y=floor(tempy);
							//	printf("%d\t%d\n",x,y);
								//abs(x-x2)>1 && abs(y-y2)>2
								if(track_box.angle<70.0 && track_box.angle>3.0)
									paintflg=1;
								else
									paintflg=0;
								if((abs(x-x2)>1 && abs(y-y2)>2) && moveflag==0 && paintflg==0)
									MoveMouse(x,y);
								if(GetAsyncKeyState(VK_NUMPAD1))	
								{
									moveflag=1;
								//	break;
								}
				  

							}	
					
							x2=x;
							y2=y;
									
						}
						 
						

					}


				}

			
			
        }
        
        if( select_object && selection.width > 0 && selection.height > 0 )
        {
            cvSetImageROI( image, selection );
            ( image, cvScalarAll(255), image, 0 );
            cvResetImageROI( image );
        }

        cvShowImage( "CamShiftDemo", image );
        cvShowImage( "Histogram", histimg );

        c = cvWaitKey(10);
        if( (char) c == 27 )
            break;
        switch( (char) c )
        {
        case 'b':
            backproject_mode ^= 1;
            break;
        case 'c':
            track_object = 0;
            cvZero( histimg );
            break;
        case 'h':
            show_hist ^= 1;
            if( !show_hist )
                cvDestroyWindow( "Histogram" );
            else
                cvNamedWindow( "Histogram", 1 );
            break;
        default:
            ;
        }
    }

    cvReleaseCapture( &capture );
    cvDestroyWindow("CamShiftDemo");

    return 0;
}

void MoveMouse(int x,int y)
{

INPUT *buffer = new INPUT[3];
(buffer+1)->type = INPUT_MOUSE;
(buffer+1)->mi.dx = x;
(buffer+1)->mi.dy = y;
(buffer+1)->mi.mouseData = 0;
(buffer+1)->mi.dwFlags = MOUSEEVENTF_ABSOLUTE;//MOUSEEVENTF_LEFTDOWN;
(buffer+1)->mi.time = 0;
(buffer+1)->mi.dwExtraInfo = 0;

(buffer+2)->type = INPUT_MOUSE;
(buffer+2)->mi.dx = x;
(buffer+2)->mi.dy = y;
(buffer+2)->mi.mouseData = 0;
(buffer+2)->mi.dwFlags = MOUSEEVENTF_ABSOLUTE;//MOUSEEVENTF_LEFTUP;
(buffer+2)->mi.time = 0;
(buffer+2)->mi.dwExtraInfo = 0;
SetCursorPos(x,y);
SendInput(3,buffer,sizeof(INPUT));
delete (buffer);

}

#ifdef _EiC
main(1,"camshiftdemo.c");
#endif

///////////////////////
//KEYBOARD FUNCTION

void gesture()
{
	void capturef();
	void color_to_gray();
	void gray_to_thresh();
	void thresh_to_edge();
	void edge_to_dilate();
	void dilate_to_erosion();
	void contour();

    keyflag=0;
	Sleep(10);
	while(1)
	{
		
		capturef();
		color_to_gray();
		gray_to_thresh();
		thresh_to_edge();
		edge_to_dilate();
		dilate_to_erosion();
		contour();
		if(keyflag==1)
			break;
    }
	capturef();
		

}
void capturef()
{
	int nframes = 25,i=0;
	IplImage *img =0;
	CvCapture *capture1 = 0;
	capture1 = cvCaptureFromCAM(CV_CAP_ANY);
	
	if(!capture1)
	{
		printf("error");
		exit(0);
	}
	
	if(!cvGrabFrame(capture1)) // capture a frame
	{
		printf("Could not grab a frame from given AVI file \n");
		exit(0);
	}
	
	cvNamedWindow("grabframe",CV_WINDOW_AUTOSIZE);
	
    for(i=0 ; i < nframes; i++)
	{
		img = cvQueryFrame(capture1);
		cvShowImage("grabframe", img );
        cvMoveWindow("grabframe", 1, 1); //newly added


			// wait for a key
		cvWaitKey(20);// say wait for 20 msec for a key to be pressed.
	}

	img=cvRetrieveFrame(capture1);  
	cvSaveImage("color.bmp",img);
	cvReleaseCapture( &capture1 );
	if(keyflag==1)
		cvDestroyWindow( "grabframe" );
}

void color_to_gray()
{
		//Conversion to gray scale
	color_img0 = cvLoadImage("color.bmp",1);
	color_img = cvCloneImage( color_img0 );

	gray_img0 = cvCreateImage( cvSize(color_img->width, color_img->height), 8, 1 );
	cvCvtColor( color_img, gray_img0, CV_BGR2GRAY );
  	gray_img = cvCloneImage( gray_img0 );
	
	cvSaveImage("gray.bmp",gray_img);
	
}

void gray_to_thresh()
{
	thresh=cvCreateImage( cvSize(color_img->width, color_img->height), 8, 1 );

	//THRESHOLDING THE IMAGE
	//cvThreshold( gray_img, thresh, (float)edge_thresh, (float)edge_thresh, CV_THRESH_BINARY );
	cvThreshold( gray_img, thresh, (float)edge_thresh, (float)edge_thresh,CV_THRESH_BINARY );
	cvSaveImage("threshold.bmp",thresh);
}

void thresh_to_edge()
{
		edgeImage = cvCreateImage(cvSize(color_img->width, color_img->height), 8, 1); //Create Image

	//Perform edge detection using the canny edge detector
	cvCanny(thresh, edgeImage, 100, 100,3);  
    
	cvSaveImage("edge_detection.bmp",edgeImage);
	cvShowImage( "image", edgeImage );
}

void edge_to_dilate()
{
		//PERFORM MORPHOLOGY
	//PERFORM DILATION FOR CONNECTING THE PIXELS	
      dilate_img = cvCreateImage(cvSize(color_img->width, color_img->height), 8, 1); //Create Image
      cvDilate( edgeImage, dilate_img,CV_SHAPE_RECT, 2);
      cvSaveImage("dilate.bmp",dilate_img);
}

void dilate_to_erosion()
{
	  //PERFORM EROSION

      erosion_img = cvCreateImage(cvSize(color_img->width, color_img->height), 8, 1);
	  cvErode (dilate_img, erosion_img, CV_SHAPE_RECT, 1);
	  cvSaveImage("erosion.bmp",erosion_img);
}




void contour()
{
	int i=0,cont_no,no_of_pixels,n,t;
    double r[6000];   //STORES CENTROID DISTANCES U:ARRAY STORES COEFFICIENT OF FOURIRE DESCRIPTER
    double utemp[5000];                            //STORES FOURIER DESCRIPTOR VALUES
    long int j=0;   ///////////
    double F[5000],st,st1,real[5000],imaginary[5000],ut[5000];
    double TEMP[5000];  //FOR STORING VALUES FROM FILE WHICH STORES FD OF A & B SHAPE
   
    double TEMPFD[50],sub,Fsubtract[50],t1[50],k1;             //STORES 1st 20 FD & LAST 20 FD IN THIS ARRAY
    int position=0,position1=0,k=0,fle=0;
	char arr1[9][9]={{"a.txt"},{"b.txt"},{"c.txt"},{"d.txt"},{"h.txt"},{"e.txt"},{"l.txt"},{"o.txt"},{"end.txt"}};
	int arr[10][10],tmp0,tmp1;
    double sub1;

	CvPoint pt[7000];
    CvPoint center ={0,0};
    IplImage* src;
  
    IplImage* src1;

    IplImage* dst;
    IplImage* dst1;
	CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contour = 0;
   
    src = cvLoadImage("erosion.bmp",1);
    src1 = cvCloneImage(src);
  
    dst = cvCreateImage( cvSize(src1->width, src1->height), 8, 1 );
  
    cvCvtColor( src1, dst, CV_BGR2GRAY );

    dst1 = cvCreateImage(cvSize(dst->width, dst->height), 8, 1); //Create Image
   

    cvCanny(dst,dst1,50,200,3);

	//FIND CONTOURS & X  Y CO - ORDINATE

    cont_no=cvFindContours(dst,storage, &contour, sizeof(CvContour),CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cvPoint(0,0));
  
    for( ; contour != 0; contour = contour->h_next )
    {
        CvSeq *result = contour;
      
        for(i=0; i<result->total; i++)
        {
          pt[i] = *(CvPoint*)cvGetSeqElem( result, i );        
          //////////////////printf("%d \t %d : %d\n", pt[i].x, pt[i].y, result->total);
          center.x =  center.x + pt[i].x;
          center.y =  center.y + pt[i].y;
          //////////////////printf("NO OF LOOPS=> %ld\n\n",j);
         
         
          j=j+1;       
         // if(j==100||j==200||j==300||j==400||j==500||j==600||j==700||j==800||j==900||j==1000||j==1100||j==1200||j==1300||j==1400||j==1479)
        //      getch();

        }
        center.x =  center.x /result->total;
        center.y =  center.y /result->total;
        no_of_pixels=result->total;  
          
     }
  
    //getch();
  ///////////  printf("%d \t %d\n",center.x,center.y);
    
    // SHAPE SIGNATURE
    //TYPE =CENTROID DISTANCE (IT IS INVARIANT TO TRANLATION
     for(i=0;i<no_of_pixels;i++)
     {
         r[i]=sqrt(((pt[i].x-center.x)*(pt[i].x-center.x))+((pt[i].y-center.y)*(pt[i].y-center.y)));
     }
     for(i=0;i<no_of_pixels;i++)
     {
         
         //////////////////////printf("%lf\n",r[i]);
    //     if(i==100||i==200||i==300||i==400||i==500||i==600||i==700||i==800||i==900||i==1000||i==1100||i==1200||i==1300||i==1400||i==1479)
    //         getch();
     }

///////////////************************///////////////////////

	   //DISCRRETE FOURIER TRANSFORM ON SHAPE SIGNATURE
     //no_of_pixels
     i=0;
       
     
     st=0;
     st1=0;

     for(n=0;n<no_of_pixels;n++)
     {
       
        for(t=0;t<no_of_pixels;t++)
        {
           
            utemp[t]=r[t]*(cos((2*M_PI*n*t)/no_of_pixels));
            st=st+utemp[t];                             //real
           
           
            ut[t]=r[t]*(sin((2*M_PI*n*t)/no_of_pixels));
            st1=st1+ut[t];                  //imaginary
       
        }
                st=st/no_of_pixels;
                st1=st1/no_of_pixels;
        real[n]=st;
        imaginary[n]=st1;
     
     }
              


    fp=fopen("avi.txt","w"); ////REMOVE THIS COMENT ONLY WHEN U WANT TO WRITE THE OF FD TO FILE



     //  MAGNITUDE OF FOURIER DESCRIPTOR
    
     //  INVARIANCE TO ROTATION

    for(n=0;n<no_of_pixels;n++)
    {
   
        F[n]=sqrt((real[n]*real[n])+(imaginary[n]*imaginary[n]));
        fprintf(fp,"%lf\n",F[n]);  //REMOVE THIS COMENT ONLY WHEN U WANT TO WRITE THE OF FD TO FILE
    }

    fclose(fp);  //REMOVE THIS COMENT ONLY WHEN U WANT TO WRITE THE OF FD TO FILE

    for(n=0;n<no_of_pixels;n++)
    {
        //////////////////////printf("%lf\n",F[n]);

    //    if(n==100||n==200||n==300||n==400||n==500||n==600||n==700||n==800||n==900||n==1000||n==1100||n==1200||n==1300||n==1400||n==1479)
    //         getch();
    }
	printf("%d\n",no_of_pixels);
   
   for(fle=0;fle<9;fle++)
   {
	   fp=fopen(arr1[fle],"r");

		for(i=0;i<20;i++)
		{
   			fscanf(fp,"%lf",&TEMP[i]);		
			TEMPFD[i]=F[i];
		}
        fseek(fp,-204,SEEK_END);
		position=ftell(fp);
		for(j=0,k=n-20;j<=20,k<=n;j++,k++)
		{
			fscanf(fp,"%lf",&TEMP[i]);       
	        k1=(int)F[k]*10000;
            k1=k1/10000;
            TEMPFD[i]=k1;
			i++;
		}
		Fsubtract[0]=0.0;
        sub1=0;
        for(i=0;i<40;i++)
		{
			if(TEMP[i]>TEMPFD[i])
			{
				sub=TEMP[i]-TEMPFD[i];
				t1[i]=sub;
			}
			else
			{
				sub=TEMPFD[i]-TEMP[i];
				t1[i]=sub;
			}
		    sub1=sub1+sub;
       
		}
		arr[fle][0]=floor(sub1);
		arr[fle][1]=fle;
        fclose(fp);
    }
   	for(i=0;i<fle;i++)
	{
		for(j=0;j<fle-1-i;j++)
		{
			if(arr[j+1][0]<arr[j][0])
			{
				tmp0=arr[j][0];
				tmp1=arr[j][1];
				arr[j][0]=arr[j+1][0];
				arr[j][1]=arr[j+1][1];
				arr[j+1][0]=tmp0;
				arr[j+1][1]=tmp1;
			}
		}
	}
	printf("\nFINAL ARRAY IS:\n");
	for(i=0;i<fle;i++)
			printf("%d\t%d\n",arr[i][0],arr[i][1]);

	if(arr[0][1]==0)
		sendchar(65);
	if(arr[0][1]==1)
		sendchar(66);
	if(arr[0][1]==2)
		sendchar(69);
	if(arr[0][1]==3)
		sendchar(68);
	if(arr[0][1]==4)
		sendchar(72);
	if(arr[0][1]==5)
		sendchar(69);
	if(arr[0][1]==6)
		sendchar(76);
	if(arr[0][1]==7)
		sendchar(79);
    if(arr[0][1]==8)
		keyflag=1; 

}


void sendchar(int ke)
{
//	if(fl==0)
//	{
//		ShellExecute(NULL, "open", "F:\Games\Grand Theft Auto Vice City\gta-vc.exe", NULL, NULL, SW_SHOWNORMAL);
//		Sleep(200);
		
//	}

if(fl==0)
{
	Sleep(20);
	HWND windowHandle = FindWindow(NULL,"Untitled - Notepad");
	SetWindowText(windowHandle, "Hand Gesture Recognition");
	SetFocus(windowHandle);
	fl=1;
}


		HWND windowHandle = FindWindow(NULL,"Hand Gesture Recognition");
	    SetWindowText(windowHandle, "Hand Gesture Recognition");
		Sleep(20);
	//	SetFocus(windowHandle);


//HWND windowHandle = FindWindow("NPclass","M4573R M0U53");

INPUT *key;
if(windowHandle == NULL)
 cout << "not found";
SetForegroundWindow(windowHandle);
Sleep(20);

key = new INPUT;
key->type = INPUT_KEYBOARD;
key->ki.wVk = ke;
key->ki.dwFlags = 0;
key->ki.time = 0;
key->ki.wScan = 0;
key->ki.dwExtraInfo = 0;
SendInput(1,key,sizeof(INPUT));
key->ki.dwExtraInfo = KEYEVENTF_KEYUP;

}
