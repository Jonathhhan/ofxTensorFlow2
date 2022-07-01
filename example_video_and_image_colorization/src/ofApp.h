#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "ofxCv.h"

// #define USE_VIDEO

class ofApp : public ofBaseApp {

public:
	void setup();
	void update();
	void draw();

	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y);
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);

	ofxTF2::Model model;
	cppflow::tensor input;
	cppflow::tensor input_resized;
	cppflow::tensor output;

#ifdef USE_VIDEO
	ofVideoPlayer videoPlayer;
#else
	ofImage imgIn;
#endif

	ofFloatImage imgOut;
	int width;
	int height;
};