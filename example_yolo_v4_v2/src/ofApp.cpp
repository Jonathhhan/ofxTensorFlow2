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
	max_element_vector.clear();
	max_element_index_vector.clear();
	boundings.clear();
	imgIn.load("kite.jpg");
	cppflow::tensor input = ofxTF2::imageToTensor(imgIn);
	input = cppflow::expand_dims(input, 0);
	input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
	input = cppflow::div(input, cppflow::tensor({ 255.f }));
	input = cppflow::resize_bicubic(input, cppflow::tensor({ 416, 416 }), true);

	cppflow::tensor output = model.runModel(input);
	std::vector<float> output_vector;
	ofxTF2::tensorToVector(output, output_vector);
	int rectangle_number = output_vector.size() / 84;
	std::vector<float> bound;
	for (int i = 0; i < rectangle_number; i++) {
		first = output_vector.begin() + 84. * i;
		last = output_vector.begin() + 84. * i + 4;
		std::vector<float> new_vec(first, last);
		boundings.push_back(new_vec);
		bound.insert(bound.end(), new_vec.begin(), new_vec.end());
		first = output_vector.begin() + 84. * i + 4;
		last = output_vector.begin() + 84. * i + 84;
		std::vector<float> new_vec_id(first, last);
		int max_element_index = std::max_element(new_vec_id.begin(), new_vec_id.end()) - new_vec_id.begin();
		float max_element = new_vec_id[max_element_index];
		max_element_index_vector.push_back(max_element_index);
		max_element_vector.push_back(max_element);
	}
	cppflow::tensor rectangle_tensor = ofxTF2::vectorToTensor(bound, ofxTF2::shapeVector{ rectangle_number, 4 });
	cppflow::tensor max_element_tensor = ofxTF2::vectorToTensor(max_element_vector);
	cppflow::tensor rectangle_index_tensor = cppflow::non_max_suppression(rectangle_tensor, max_element_tensor, 10, 0.5);
	ofxTF2::tensorToVector(rectangle_index_tensor, rectangle_index);
#endif
}

//--------------------------------------------------------------
void ofApp::update() {
#ifdef USE_VIDEO
	videoPlayer.update();
	if (videoPlayer.isFrameNew()) {
		max_element_vector.clear();
		max_element_index_vector.clear();
		boundings.clear();
		cppflow::tensor input = ofxTF2::pixelsToTensor(videoPlayer.getPixels());
		input = cppflow::expand_dims(input, 0);
		input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
		input = cppflow::div(input, cppflow::tensor({ 255.f }));
		input = cppflow::resize_bicubic(input, cppflow::tensor({ 416, 416 }), true);

		cppflow::tensor output = model.runModel(input);
		std::vector<float> output_vector;
		ofxTF2::tensorToVector(output, output_vector);
		int rectangle_number = output_vector.size() / 84;
		std::vector<float> bound;
		for (int i = 0; i < rectangle_number; i++) {
			first = output_vector.begin() + 84. * i;
			last = output_vector.begin() + 84. * i + 4;
			std::vector<float> new_vec(first, last);
			boundings.push_back(new_vec);
			bound.insert(bound.end(), new_vec.begin(), new_vec.end());
			first = output_vector.begin() + 84. * i + 4;
			last = output_vector.begin() + 84. * i + 84;
			std::vector<float> new_vec_id(first, last);
			int max_element_index = std::max_element(new_vec_id.begin(), new_vec_id.end()) - new_vec_id.begin();
			float max_element = new_vec_id[max_element_index];
			max_element_index_vector.push_back(max_element_index);
			max_element_vector.push_back(max_element);
		}
		cppflow::tensor rectangle_tensor = ofxTF2::vectorToTensor(bound, ofxTF2::shapeVector{ rectangle_number, 4 });
		cppflow::tensor max_element_tensor = ofxTF2::vectorToTensor(max_element_vector);
		cppflow::tensor rectangle_index_tensor = cppflow::non_max_suppression(rectangle_tensor, max_element_tensor, 10, 0.5);
		ofxTF2::tensorToVector(rectangle_index_tensor, rectangle_index);
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

	
	for (int i = 0; i < rectangle_index.size(); i++) {
		int index = rectangle_index[i];
		if (max_element_vector[index] > 0.2) {
			if (max_element_index_vector[index] == 0) {
				ofSetColor(0, 0, 255);
			}
			else if (max_element_index_vector[index] == 2) {
				ofSetColor(0, 255, 0);
			}
			else {
				ofSetColor(255, 0, 0);
			}
			ofDrawRectangle(boundings[index][1] * 480 + 20, boundings[index][0] * 360 + 20, boundings[index][3] * 480 - boundings[index][1] * 480, boundings[index][2] * 360 - boundings[index][0] * 360);
			ofDrawBitmapStringHighlight("id: " + cocoClasses[max_element_index_vector[index]] + ", prob: " + ofToString(max_element_vector[index]), boundings[index][1] * 480 + 30, boundings[index][0] * 360 + 40);
		}
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