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
	imgIn2.allocate(videoPlayer.getWidth(), videoPlayer.getHeight(), OF_IMAGE_COLOR);
	videoPlayer.play();
#else
	maxElementVector.clear();
	maxElementIndexVector.clear();
	boundings.clear();
	rectangleIndex.clear();
	imgIn.load("eisenstein.jpg");
	imgIn2.allocate(imgIn.getWidth(), imgIn.getHeight(), OF_IMAGE_COLOR);
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
	cppflow::tensor te3 = cppflow::non_max_suppression(te1, te2, 10, 0.5);
	ofxTF2::tensorToVector(te3, rectangleIndex);
	cppflow::tensor te4 = cppflow::gather(te1, te3, TF_FLOAT, TF_FLOAT);
	te4 = cppflow::expand_dims(te4, 0);
	te4 = cppflow::cast(te4, TF_UINT8, TF_FLOAT);
	cppflow::tensor te5 = cppflow::tensor({ 1.0, 0.2, 0.0 });
	te5 = cppflow::expand_dims(te5, 0);
	te5 = cppflow::cast(te5, TF_UINT8, TF_FLOAT);
	input = cppflow::draw_bounding_boxes_v2(input, te4, te5);
	ofxTF2::tensorToImage(input, imgIn2);
	imgIn2.update();
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
		cppflow::tensor te3 = cppflow::non_max_suppression(te1, te2, 10, 0.5);
		ofxTF2::tensorToVector(te3, rectangleIndex);
		cppflow::tensor te4 = cppflow::gather(te1, te3, TF_FLOAT, TF_FLOAT);
		te4 = cppflow::expand_dims(te4, 0);
		te4 = cppflow::cast(te4, TF_UINT8, TF_FLOAT);
		cppflow::tensor te5 = cppflow::tensor({ 1.0, 0.2, 0.0 });
		te5 = cppflow::expand_dims(te5, 0);
		te5 = cppflow::cast(te5, TF_UINT8, TF_FLOAT);
		input = cppflow::draw_bounding_boxes_v2(input, te4, te5);
		ofxTF2::tensorToImage(input, imgIn2);
		imgIn2.update();
	}
#endif
}

//--------------------------------------------------------------
void ofApp::draw() {
	imgIn2.draw(20, 20, 480, 360);
	for (int i = 0; i < rectangleIndex.size(); i++) {
		int index = rectangleIndex[i];
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