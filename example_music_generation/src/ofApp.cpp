#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("example_music_generation");
	ofSetBackgroundColor(150, 200, 200);
	sucessTime = 0;
	note_length = 0;

	midiOut.listOutPorts();
	midiOut.openPort(0);
	channel = 1;
	currentPgm = 0;
	midiOut.sendProgramChange(channel, currentPgm);

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}

	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}

	model.setup({ "serving_default_input_1" }, { "StatefulPartitionedCall:1", "StatefulPartitionedCall:0", "StatefulPartitionedCall:2" });
	vect = { 88, 1.5, 3.5, 55, 0.5, 2.5, 60, 0.5, 2.5, 50, 3.5, 5.5, 60, 0.5, 1.5, 38, 1.5, 4.5, 55, 0.5, 7.5, 60, 1.5, 5.5, 50, 2.5, 3.5, 60, 3.5, 0.5, 38, 0.5, 0.5, 55, 0.5, 0.5, 60, 0.5, 0.5, 50, 0.5, 0.5, 60, 0.5, 0.5, 38, 0.5, 0.5, 55, 0.5, 6.5, 60, 0.5, 6.5, 50, 0.5, 3.5, 60, 0.5, 8.5, 38, 0.5, 4.5, 55, 0.5, 5.5, 60, 1.5, 3.5, 50, 1.5, 7.5, 60, 3.5, 4.5 };
	t = ofxTF2::vectorToTensor(vect);
	t = cppflow::reshape(t, { 25, 3 }, TF_FLOAT);
	t = cppflow::expand_dims(t, 0);
	t = cppflow::div(t, { 128.f, 1.f, 1.f });
	t = cppflow::cast(t, TF_INT32, TF_FLOAT);
}

//--------------------------------------------------------------
void ofApp::update() {
	actualTime = ofGetElapsedTimeMillis();
	if (actualTime > sucessTime) {	
		vector <cppflow::tensor> output = model.runMultiModel({ t });
		t = cppflow::multinomial(output[0], 1);
		pitch = t.get_data<int64_t>()[0];
		step = output[1].get_data<float>()[0];
		duration = output[2].get_data<float>()[0];
		cout << "pitch: " << ofToString(pitch) << endl;
		cout << "step: " << ofToString(step) << endl;
		cout << "duration: " << ofToString(duration) << endl;
		vect.erase(vect.begin(), vect.begin() + 3);
		vect.push_back(pitch);
		vect.push_back(step);
		vect.push_back(duration);
		t = ofxTF2::vectorToTensor(vect);
		t = cppflow::reshape(t, { 25, 3 }, TF_FLOAT);
		t = cppflow::expand_dims(t, 0);
		t = cppflow::div(t, { 128.f, 1.f, 1.f });
		t = cppflow::cast(t, TF_INT32, TF_FLOAT);
		midiOut.sendNoteOn(channel, pitch, 70);
		sucessTime = actualTime + step * 500;
		note_length = actualTime + duration * 1000;
	}
	if (actualTime > note_length) {
		midiOut.sendNoteOff(channel, pitch, 70);
	}
}

//--------------------------------------------------------------
void ofApp::draw() {

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
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}