#include "ofApp.h"

tokenizers::SubwordTextEncoder textEncoder("data/tokenizer.tf");

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("example_chatbot");
	ofSetBackgroundColor(150, 200, 200);

	_textParameter.addListener(this, &ofApp::onTextChange);
	_parameters.setName("Text panel");
	_parameters.add(_textParameter.set("Type something", "Default"));
	_gui.setTextColor(240);
	_gui.setDefaultTextColor(240);
	_gui.setDefaultWidth(500);
	_gui.setup(_parameters);
	_gui.setPosition(350, 50);

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
	ofSetColor(240);
	ofDrawBitmapString("You: " + decoded_question, 50, 200);
	ofDrawBitmapString("The bot: " + decoded_answer, 50, 250);
	_gui.draw();
}

void ofApp::onTextChange(std::string& text) {
	if (model.isLoaded()) {
		int maxElementIndex = 0;
		std::vector<int> input_vector = textEncoder.encode(text);
		input_vector.insert(input_vector.begin(), vocabSize + 257);
		input_vector.push_back(vocabSize + 258);
		cppflow::tensor input_1 = ofxTF2::vectorToTensor(input_vector);
		input_1 = cppflow::expand_dims(input_1, 0);
		input_1 = cppflow::cast(input_1, TF_INT32, TF_FLOAT);
		std::vector<int> output_vector = { vocabSize + 257 };
		cppflow::tensor input_2 = ofxTF2::vectorToTensor(output_vector);
		input_2 = cppflow::expand_dims(input_2, 0);
		input_2 = cppflow::cast(input_2, TF_INT32, TF_FLOAT);

		for (int i = 0; i < 40; i++) {
			if (maxElementIndex == textEncoder.get_vocab_size() + 258) {
				break;
			}
			std::vector<cppflow::tensor> vectorOfInputTensors = { input_1, input_2 };
			std::vector<cppflow::tensor> vectorOfOutputTensors = model.runMultiModel(vectorOfInputTensors);
			vectorOfOutputTensors[0] = cppflow::slice(vectorOfOutputTensors[0], cppflow::tensor({ 0, i, 0 }), cppflow::tensor({ 1, 1, -1 }), cppflow::datatype(TF_FLOAT));
			cppflow::tensor max = cppflow::arg_max(vectorOfOutputTensors[0], 2);
			maxElementIndex = max.get_data<int64_t>()[0];
			output_vector.push_back(maxElementIndex);
			max = cppflow::cast(max, TF_INT32, TF_FLOAT);
			input_2 = cppflow::concat(1, { input_2, max });
			input_2 = cppflow::cast(input_2, TF_INT32, TF_FLOAT);
		}

		input_vector.pop_back();
		input_vector.erase(input_vector.begin());
		decoded_question = textEncoder.decode(input_vector);
		ofStringReplace(decoded_question, "_", " ");
		decoded_question = std::regex_replace(decoded_question, std::regex("\\s+"), " ");
		decoded_question = std::regex_replace(decoded_question, std::regex("\\s([,.!?])"), "$1");
		std::cout << "Decoded question: " << decoded_question << std::endl;
		std::cout << "Encoded question: " << ofToString(input_vector) << std::endl;

		output_vector.pop_back();
		output_vector.erase(output_vector.begin());
		decoded_answer = textEncoder.decode(output_vector);
		ofStringReplace(decoded_answer, "_", " ");
		decoded_answer = std::regex_replace(decoded_answer, std::regex("\\s+"), " ");
		decoded_answer = std::regex_replace(decoded_answer, std::regex("\\s([,.!?])"), "$1");
		std::cout << "Decoded answer: " << decoded_answer << std::endl;
		std::cout << "Encoded answer: " << ofToString(output_vector) << std::endl;
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