#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"
#include "Tokenizers.h"
#include "ofxGui.h"

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

		void onTextChange(std::string& text);

		ofxTF2::Model model;
		ofxPanel _gui;
		ofParameterGroup _parameters;
		ofParameter<std::string> _textParameter;
		ofEventListener _textParameterListener;

		std::string decoded_question;
		std::string decoded_answer;
		int vocabSize;
};
