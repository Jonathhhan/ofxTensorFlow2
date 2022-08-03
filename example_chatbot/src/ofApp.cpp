#include "ofApp.h"

tokenizers::SubwordTextEncoder textEncoder("data/tokenizer.tf");

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("example_chatbot");

	_textParameter.addListener(this, &ofApp::onTextChange);
	_parameters.setName("Type something:");
	_parameters.add(_textParameter.set("text", "default"));
	_gui.setDefaultWidth(400);
	_gui.setup(_parameters);

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}

	if (!model.load("save_model")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({ "serving_default_inputs", "serving_default_dec_inputs" }, { "StatefulPartitionedCall" });

	vocabSize = textEncoder.get_vocab_size();
	std::cout << "Size of vocabulary " << vocabSize << std::endl;
}

//--------------------------------------------------------------
void ofApp::update() {

}

//--------------------------------------------------------------
void ofApp::draw() {
	ofSetColor(20);
	ofDrawRectangle(20, 170, 1160, 100);
	ofSetColor(200);
	ofDrawBitmapString("Question: " + decoded_question, 50, 200);
	ofDrawBitmapString("Answer: " + decoded_answer, 50, 250);
	_gui.draw();
}

void ofApp::onTextChange(std::string& text) {
	// ofLogNotice() << "text changed " << text;
	if (model.isLoaded()) {
		int maxElementIndex = 0;
		std::list<int> encoded_words_1 = textEncoder.encode(text);
		std::vector<float> tempVector_1(encoded_words_1.begin(), encoded_words_1.end());
		tempVector_1.insert(tempVector_1.begin(), vocabSize + 256);
		tempVector_1.push_back(vocabSize + 257);
		cppflow::tensor input_1 = ofxTF2::vectorToTensor(tempVector_1);
		cppflow::tensor input_2 = cppflow::tensor({ vocabSize + 256 });
		input_1 = cppflow::expand_dims(input_1, 0);
		input_2 = cppflow::expand_dims(input_2, 0);
		input_1 = cppflow::cast(input_1, TF_INT32, TF_FLOAT);
		input_2 = cppflow::cast(input_2, TF_INT32, TF_FLOAT);
		for (int i = 0; i < 40; i++) {
			if (maxElementIndex == textEncoder.get_vocab_size() + 257) {
				break;
			}
			std::vector<cppflow::tensor> vett = { input_1, input_2 };
			std::vector<cppflow::tensor> output;
			output = model.runMultiModel(vett);
			ofxTF2::tensorToVector(output[0], tempVector_1);
			vector<float> tempVector_2(tempVector_1.begin() + 8278 * i, tempVector_1.end());
			maxElementIndex = std::max_element(tempVector_2.begin(), tempVector_2.end()) - tempVector_2.begin();
			ofxTF2::tensorToVector(input_2, tempVector_1);
			tempVector_1.push_back((float)maxElementIndex);
			input_2 = ofxTF2::vectorToTensor(tempVector_1);
			input_2 = cppflow::expand_dims(input_2, 0);
		}

		decoded_question = textEncoder.decode(encoded_words_1);
		ofStringReplace(decoded_question, "_", " ");
		std::cout << "Decoded question: " << decoded_question << std::endl;
		std::cout << "Encoded question: ";
		for (auto& word : encoded_words_1) {
			std::cout << word << " ";
		}
		std::cout << endl;

		ofxTF2::tensorToVector(input_2, tempVector_1);
		tempVector_1.pop_back();
		tempVector_1.erase(tempVector_1.begin());
		std::list<int> encoded_words_2(tempVector_1.begin(), tempVector_1.end());
		decoded_answer = textEncoder.decode(encoded_words_2);
		ofStringReplace(decoded_answer, "_", " ");
		decoded_answer = std::regex_replace(decoded_answer, std::regex(" +"), " ");
		std::cout << "Decoded answer: " << decoded_answer << std::endl;
		std::cout << "Encoded answer: ";
		for (auto& word : encoded_words_2) {
			std::cout << word << " ";
		}
		std::cout << endl;
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
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}