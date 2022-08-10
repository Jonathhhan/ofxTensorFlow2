#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("example_music_generation");
	ofSetBackgroundColor(150, 200, 200);

	// print the available output ports to the console
	midiOut.listOutPorts();

	// connect
	midiOut.openPort(0); // by number
	//midiOut.openPort("IAC Driver Pure Data In"); // by name
	//midiOut.openVirtualPort("ofxMidiOut"); // open a virtual port

	channel = 1;
	currentPgm = 46;
	note = 0;
	velocity = 0;
	pan = 0;
	bend = 0;
	touch = 0;
	polytouch = 0;

	midiOut.sendProgramChange(channel, currentPgm);

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}

	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}

	model.setup({ "serving_default_input_1" }, { "StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2" });
	vect = { 88, 0.5, 0.5, 55, 0.5, 0.5, 60, 0.5, 0.5, 50, 0.5, 0.5, 60, 0.5, 0.5, 38, 0.5, 0.5, 55, 0.5, 0.5, 60, 0.5, 0.5, 50, 0.5, 0.5, 60, 0.5, 0.5, 38, 0.5, 0.5, 55, 0.5, 0.5, 60, 0.5, 0.5, 50, 0.5, 0.5, 60, 0.5, 0.5, 38, 0.5, 0.5, 55, 0.5, 0.5, 60, 0.5, 0.5, 50, 0.5, 0.5, 60, 0.5, 0.5, 38, 0.5, 0.5, 55, 0.5, 0.5, 60, 0.5, 0.5, 50, 0.5, 0.5, 60, 0.5, 0.5 };
	t = ofxTF2::vectorToTensor(vect);
	
	cout << ofToString(t.shape()) << endl;
	t = cppflow::reshape(t, { 25, 3 }, TF_FLOAT);
	t = cppflow::expand_dims(t, 0);
	t = cppflow::div(t, { 128.f, 1.f, 1.f });
	t = cppflow::cast(t, TF_INT32, TF_FLOAT);
	
	sucessTime = 0;
}

//--------------------------------------------------------------
void ofApp::update() {
	actualTime = ofGetElapsedTimeMillis();
	if (actualTime > sucessTime) {
		
		vector <cppflow::tensor> output = model.runMultiModel({ t });
		output[1] = cppflow::add(output[1], 128.f);
		t = ofxTF2::vectorToTensor(vect);
		std::discrete_distribution<> d(vect.begin(), vect.end());
		std::random_device rd;
		std::mt19937 gen(rd());
		int abc = d(gen) + 24;
		cout << "pitch: " << ofToString(abc) << endl;
		cout << "step: " << ofToString(output[2].get_data<float>()) << endl;
		cout << "duration: " << ofToString(output[0].get_data<float>()) << endl;
		vect.erase(vect.begin(), vect.begin() + 3);
		vect.push_back(abc);
		vect.push_back(output[2].get_data<float>()[0]);
		vect.push_back(output[0].get_data<float>()[0]);
		t = ofxTF2::vectorToTensor(vect);
		t = cppflow::reshape(t, { 25, 3 }, TF_FLOAT);
		t = cppflow::expand_dims(t, 0);
		t = cppflow::cast(t, TF_INT32, TF_FLOAT);
		midiOut.sendNoteOn(channel, abc, 70);
		sucessTime = actualTime + output[2].get_data<float>()[0] * 1000;
		// print out both the midi note and the frequency
		ofLogNotice() << "note: " << abc
			<< " freq: " << ofxMidi::mtof(abc) << " Hz";
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