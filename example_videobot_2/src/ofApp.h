#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "srtparser.h"
#include "Tokenizers.h"

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
		std::string chatbot(std::string str);

		ofVideoPlayer videoPlayer;
		ofxTF2::Model model;
		ofxTF2::Model bot;
		std::vector<SubtitleItem*> sub;
		std::vector<std::tuple<std::vector<double>, int, int, std::vector<double>>> vector_sub;
		std::vector<std::tuple<std::vector<double>, int, int, std::vector<double>>> vector_sub_copy;
		std::vector<double> nextVector;
		int currentSubNo;
		int currentSubLenght;
		std::string currentString;
		std::string show;
		std::string decoded_question;
		std::string decoded_answer;
		int vocabSize;
};
