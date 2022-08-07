#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofBuffer buffer = ofBufferFromFile("cocoClasses.txt");
	for (auto& line : buffer.getLines()) {
		cocoClasses.push_back(line);
	}
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofSetWindowTitle("example_yolo_v4");
	ofNoFill();

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}

	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({ "serving_default_input_1" }, { "StatefulPartitionedCall" });

#ifdef USE_VIDEO
	videoPlayer.load("Frenzy.mp4");
	videoPlayer.play();
#else
	maxElementVector.clear();
	maxElementIndexVector.clear();
	boundings.clear();
	rectangleIndex.clear();
	imgIn.load("eisenstein.jpg");
	input = ofxTF2::imageToTensor(imgIn);
	input = cppflow::expand_dims(input, 0);
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::div(input, cppflow::tensor({ 255.f }));
	input_resized = cppflow::resize_bicubic(input, cppflow::tensor({ 416, 416 }), true);

	output = model.runModel(input_resized);
	ofxTF2::tensorToVector(output, vec);
	int rectangle_number = vec.size() / 84;
	std::vector<float> bound;
	for (int i = 0; i < rectangle_number; i++) {
		first = vec.begin() + 84. * i;
		last = vec.begin() + 84. * i + 4;
		std::vector<float> newVec(first, last);
		boundings.push_back(newVec);
		bound.insert(bound.end(), newVec.begin(), newVec.end());
		first = vec.begin() + 84. * i + 4;
		last = vec.begin() + 84. * i + 84;
		std::vector<float> newVecId(first, last);
		int maxElementIndex = max_element(newVecId.begin(), newVecId.end()) - newVecId.begin();
		float maxElement = newVecId[maxElementIndex];
		maxElementIndexVector.push_back(maxElementIndex);
		maxElementVector.push_back(maxElement);
}
	cppflow::tensor te1 = ofxTF2::vectorToTensor(bound, ofxTF2::shapeVector{ rectangle_number, 4 });
	cppflow::tensor te2 = ofxTF2::vectorToTensor(maxElementVector);
	cppflow::tensor te3 = cppflow::non_max_suppression(te1, te2, 10);
	ofxTF2::tensorToVector(te3, rectangleIndex);
#endif
}

//--------------------------------------------------------------
void ofApp::update() {
#ifdef USE_VIDEO
	videoPlayer.update();
	if (videoPlayer.isFrameNew()) {
		maxElementVector.clear();
		maxElementIndexVector.clear();
		boundings.clear();
		rectangleIndex.clear();
		input = ofxTF2::pixelsToTensor(videoPlayer.getPixels());
		input = cppflow::expand_dims(input, 0);
		input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
		input = cppflow::div(input, cppflow::tensor({ 255.f }));
		input_resized = cppflow::resize_bicubic(input, cppflow::tensor({ 416, 416 }), true);

		output = model.runModel(input_resized);
		ofxTF2::tensorToVector(output, vec);
		int rectangle_number = vec.size() / 84;
		std::vector<float> bound;
		for (int i = 0; i < rectangle_number; i++) {
			first = vec.begin() + 84. * i;
			last = vec.begin() + 84. * i + 4;
			std::vector<float> newVec(first, last);
			boundings.push_back(newVec);
			bound.insert(bound.end(), newVec.begin(), newVec.end());
			first = vec.begin() + 84. * i + 4;
			last = vec.begin() + 84. * i + 84;
			std::vector<float> newVecId(first, last);
			int maxElementIndex = max_element(newVecId.begin(), newVecId.end()) - newVecId.begin();
			float maxElement = newVecId[maxElementIndex];
			maxElementIndexVector.push_back(maxElementIndex);
			maxElementVector.push_back(maxElement);
		}
		cppflow::tensor te1 = ofxTF2::vectorToTensor(bound, ofxTF2::shapeVector{ rectangle_number, 4 });
		cppflow::tensor te2 = ofxTF2::vectorToTensor(maxElementVector);
		cppflow::tensor te3 = cppflow::non_max_suppression(te1, te2, 10);
		ofxTF2::tensorToVector(te3, rectangleIndex);
	}
#endif
}

//--------------------------------------------------------------
void ofApp::draw() {
	ofSetColor(255);
#ifdef USE_VIDEO
	videoPlayer.draw(20, 20, 480, 360);
#else
	imgIn.draw(20, 20, 480, 360);
#endif
	ofSetColor(255, 0, 0);
	for (int i = 0; i < rectangleIndex.size(); i++) {
		int index = rectangleIndex[i];
		if(maxElementVector[index] >= 0.2)
		ofDrawRectangle(boundings[index][1] * 480 + 20, boundings[index][0] * 360 + 20, boundings[index][3] * 480 - boundings[index][1] * 480, boundings[index][2] * 360 - boundings[index][0] * 360);
		ofDrawBitmapStringHighlight("id: " + ofToString(cocoClasses[maxElementIndexVector[index]]) + ", prob: " + ofToString(maxElementVector[index]), boundings[index][1] * 480 + 30, boundings[index][0] * 360 + 40);
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}