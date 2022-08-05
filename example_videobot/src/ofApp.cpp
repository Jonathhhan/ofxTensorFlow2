#include "ofApp.h"

tokenizers::SubwordTextEncoder textEncoder("data/tokenizer.tf");

//--------------------------------------------------------------
void ofApp::setup() {
	cppflow::tensor tensor;
	std::vector<double> vec;
	std::vector<std::string> dialogue;
	int counter = -1;

	ofSetWindowTitle("example_universal_sentence_encoder");
	videoPlayer.load("Frenzy.mp4");

	vocabSize = textEncoder.get_vocab_size();
	std::cout << "Size of vocabulary " << vocabSize << std::endl;

	if (!ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70, true)) {
		ofLogError() << "failed to set GPU Memory options!";
	}

	if (!model.load("model")) {
		std::exit(EXIT_FAILURE);
	}
	model.setup({ {"serving_default_inputs:0"} }, { {"StatefulPartitionedCall_1:0"} });

	if (!bot.load("save_model")) {
		std::exit(EXIT_FAILURE);
	}
	bot.setup({ "serving_default_inputs", "serving_default_dec_inputs" }, { "StatefulPartitionedCall" });

	SubtitleParserFactory* subParserFactory = new SubtitleParserFactory(ofToDataPath("Frenzy.srt"));
	SubtitleParser* parser = subParserFactory->getParser();
	sub = parser->getSubtitles();
	for (auto element : sub) {
		dialogue.push_back(element->getDialogue());
		if (element->getDialogue().back() == '.' || element->getDialogue().back() == '?' || element->getDialogue().back() == '!' || element->getDialogue().back() == '"' || element->getDialogue().back() == '\'' || element->getDialogue().back() == ';') {
			std::string currentDialogue = ofJoinString(dialogue, " ");
			dialogue.clear();
			tensor = model.runModel(cppflow::reshape(cppflow::tensor(currentDialogue), { -1 }));
			ofxTF2::tensorToVector(tensor, vec);
			vector_sub.push_back(std::make_tuple(vec, element->getSubNo() - counter, counter, currentDialogue));
			counter = -1;
		}
		counter++;
	}
	vector_sub_copy = vector_sub;
	std::cout << "Subtitles loaded." << std::endl;
	currentSubNo = std::get<1>(vector_sub_copy[0]);
	currentSubLenght = std::get<2>(vector_sub_copy[0]);
	currentString = std::get<3>(vector_sub_copy[0]);
	vector_sub_copy.erase(vector_sub_copy.begin());
	videoPlayer.play();
}

//--------------------------------------------------------------
void ofApp::update() {
	videoPlayer.update();
	if (currentSubNo + currentSubLenght < sub.size() && sub[currentSubNo - 1. + currentSubLenght]->getEndTime() + ((sub[currentSubNo + currentSubLenght]->getStartTime() - sub[currentSubNo - 1. + currentSubLenght]->getEndTime()) / 2.)+100 < videoPlayer.getPosition() * videoPlayer.getDuration() * 1000 ||  videoPlayer.getIsMovieDone()) {
		std::vector<double> newVector = chatbot(currentString);
		std::vector<double> cosine;
		for (int x = 0; x < vector_sub_copy.size(); x++) {
			double cosine_similarity = 0;
			for (int i = 0; i < std::get<0>(vector_sub_copy[x]).size(); i++) {
				cosine_similarity += newVector[i] * std::get<0>(vector_sub_copy[x])[i];
			}
			cosine.push_back(cosine_similarity);
		}
		int maxElementIndex = std::max_element(cosine.begin(), cosine.end()) - cosine.begin();
		double maxElement = *std::max_element(cosine.begin(), cosine.end());
		currentSubNo = std::get<1>(vector_sub_copy[maxElementIndex]);
		currentSubLenght = std::get<2>(vector_sub_copy[maxElementIndex]);
		currentString = std::get<3>(vector_sub_copy[maxElementIndex]);
		show = "Subtitle: " + ofToString(currentSubNo) + " - " + ofToString(currentSubNo + currentSubLenght) + ".\n\nCosine similarity: " + ofToString(maxElement) + ".\n\nSubtitles left: " + ofToString(vector_sub_copy.size() - 1) + ".";
		vector_sub_copy.erase(vector_sub_copy.begin() + maxElementIndex);
		if (vector_sub_copy.size() < 1) {
			vector_sub_copy = vector_sub;
		}
		if (currentSubNo > 1) {
			videoPlayer.setPosition((sub[currentSubNo - 2.]->getEndTime() + ((sub[currentSubNo - 1.]->getStartTime() - sub[currentSubNo - 2.]->getEndTime()) / 2.)) / videoPlayer.getDuration() / 1000);
		}
		else {
			videoPlayer.setPosition(0);
		}
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	videoPlayer.draw(450, 30, 300, 200);
	ofDrawBitmapStringHighlight("Press a key!", 50, 50);
	ofDrawBitmapString(show, 50,100);
	for (int i = currentSubNo - 1; i <= currentSubNo - 1 + currentSubLenght; i++) {
		if (sub[i]->getStartTime() < videoPlayer.getPosition() * videoPlayer.getDuration() * 1000 && sub[i]->getEndTime() > videoPlayer.getPosition() * videoPlayer.getDuration() * 1000) {
			ofDrawBitmapString(sub[i]->getDialogue(), 460, 250);
		}
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	std::vector<double> newVector = chatbot(currentString);
	std::vector<double> cosine;
	for (int x = 0; x < vector_sub_copy.size(); x++) {
		double cosine_similarity = 0;
		for (int i = 0; i < std::get<0>(vector_sub_copy[x]).size(); i++) {
			cosine_similarity += newVector[i] * std::get<0>(vector_sub_copy[x])[i];
		}
		cosine.push_back(cosine_similarity);
	}
	int maxElementIndex = std::max_element(cosine.begin(), cosine.end()) - cosine.begin();
	double maxElement = *std::max_element(cosine.begin(), cosine.end());
	currentSubNo = std::get<1>(vector_sub_copy[maxElementIndex]);
	currentSubLenght = std::get<2>(vector_sub_copy[maxElementIndex]);
	currentString = std::get<3>(vector_sub_copy[maxElementIndex]);
	show = "Subtitle: " + ofToString(currentSubNo) + " - " + ofToString(currentSubNo + currentSubLenght) + ".\n\nCosine similarity: " + ofToString(maxElement) + ".\n\nSubtitles left: " + ofToString(vector_sub_copy.size() - 1) + ".";
	vector_sub_copy.erase(vector_sub_copy.begin() + maxElementIndex);
	if (vector_sub_copy.size() < 1) {
		vector_sub_copy = vector_sub;
	}
	if (currentSubNo > 1) {
		videoPlayer.setPosition((sub[currentSubNo - 2.]->getEndTime() + ((sub[currentSubNo - 1.]->getStartTime() - sub[currentSubNo - 2.]->getEndTime()) / 2.)) / videoPlayer.getDuration() / 1000);
	}
	else {
		videoPlayer.setPosition(0);
	}
}

//--------------------------------------------------------------
std::vector<double> ofApp::chatbot(std::string str) {
		int maxElementIndex = 0;
		std::list<int> encoded_words_1 = textEncoder.encode(str);
		std::vector<float> tempVector_1(encoded_words_1.begin(), encoded_words_1.end());
		tempVector_1.insert(tempVector_1.begin(), vocabSize + 257);
		tempVector_1.push_back(vocabSize + 258);
		cppflow::tensor input_1 = ofxTF2::vectorToTensor(tempVector_1);
		cppflow::tensor input_2 = cppflow::tensor({ vocabSize + 257 });
		input_1 = cppflow::expand_dims(input_1, 0);
		input_2 = cppflow::expand_dims(input_2, 0);
		input_1 = cppflow::cast(input_1, TF_INT32, TF_FLOAT);
		input_2 = cppflow::cast(input_2, TF_INT32, TF_FLOAT);

		for (int i = 0; i < 40; i++) {
			if (maxElementIndex == textEncoder.get_vocab_size() + 258) {
				break;
			}
			std::vector<cppflow::tensor> vectorOfInputTensors = { input_1, input_2 };
			std::vector<cppflow::tensor> vectorOfOutputTensors = bot.runMultiModel(vectorOfInputTensors);
			ofxTF2::tensorToVector(vectorOfOutputTensors[0], tempVector_1);
			ofxTF2::tensorToVector(vectorOfOutputTensors[0], tempVector_1);
			vector<int> tempVector_3;
			ofxTF2::tensorToVector(vectorOfOutputTensors[0].shape(), tempVector_3);
			vector<float> tempVector_2(tempVector_1.begin() + tempVector_3[2] * i, tempVector_1.end());
			maxElementIndex = std::max_element(tempVector_2.begin(), tempVector_2.end()) - tempVector_2.begin();
			ofxTF2::tensorToVector(input_2, tempVector_1);
			tempVector_1.push_back((float)maxElementIndex);
			input_2 = ofxTF2::vectorToTensor(tempVector_1);
			input_2 = cppflow::expand_dims(input_2, 0);
		}

		decoded_question = textEncoder.decode(encoded_words_1);
		ofStringReplace(decoded_question, "_", " ");
		decoded_question = std::regex_replace(decoded_question, std::regex(" +"), " ");
		decoded_question = std::regex_replace(decoded_question, std::regex(" *\\."), ".");
		decoded_question = std::regex_replace(decoded_question, std::regex(" *,"), ",");
		decoded_question = std::regex_replace(decoded_question, std::regex(" *!"), "!");
		decoded_question = std::regex_replace(decoded_question, std::regex(" *\\?"), "?");
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
		decoded_answer = std::regex_replace(decoded_answer, std::regex(" *\\."), ".");
		decoded_answer = std::regex_replace(decoded_answer, std::regex(" *,"), ",");
		decoded_answer = std::regex_replace(decoded_answer, std::regex(" *!"), "!");
		decoded_answer = std::regex_replace(decoded_answer, std::regex(" *\\?"), "?");
		std::cout << "Decoded answer: " << decoded_answer << std::endl;
		std::cout << "Encoded answer: ";
		for (auto& word : encoded_words_2) {
			std::cout << word << " ";
		}
		std::cout << endl;
		cppflow::tensor output = model.runModel(cppflow::reshape(cppflow::tensor(decoded_answer), { -1 }));
		std::vector<double> vec;
		ofxTF2::tensorToVector(output, vec);
		return vec;
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