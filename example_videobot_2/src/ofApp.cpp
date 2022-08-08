#include "ofApp.h"

tokenizers::SubwordTextEncoder textEncoder("data/tokenizer.tf");

//--------------------------------------------------------------
void ofApp::setup() {
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
	model.setup({ "serving_default_inputs:0" }, { "StatefulPartitionedCall_1:0" });

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
			cppflow::tensor output_1 = model.runModel(cppflow::reshape(cppflow::tensor(currentDialogue), { -1 }));
			cppflow::tensor output_2 = model.runModel(cppflow::reshape(cppflow::tensor(chatbot(currentDialogue)), { -1 }));
			vector_sub.push_back(std::make_tuple(output_1, element->getSubNo() - counter, counter, output_2));
			counter = -1;
		}
		counter++;
	}
	vector_sub_copy = vector_sub;
	std::cout << "Subtitles loaded." << std::endl;
	currentSubNo = std::get<1>(vector_sub_copy[0]);
	currentSubLenght = std::get<2>(vector_sub_copy[0]);
	nextVector = std::get<3>(vector_sub_copy[0]);
	vector_sub_copy.erase(vector_sub_copy.begin());
	videoPlayer.play();
}

//--------------------------------------------------------------
void ofApp::update() {
	videoPlayer.update();
	if ((float)currentSubNo + currentSubLenght < sub.size() && sub[currentSubNo - 1. + currentSubLenght]->getEndTime() + ((sub[currentSubNo + currentSubLenght]->getStartTime() - sub[currentSubNo - 1. + currentSubLenght]->getEndTime()) / 2.) < videoPlayer.getPosition() * videoPlayer.getDuration() * 1000 ||  videoPlayer.getIsMovieDone()) {
		std::vector<float> cosine;
		for (auto& element : vector_sub_copy) {
			cppflow::tensor cosine_similarity = cppflow::sum(nextVector * std::get<0>(element), cppflow::tensor({ 1 }));
			cosine.push_back(cosine_similarity.get_data<float>()[0]);
		}
		cppflow::tensor cosine_tensor = ofxTF2::vectorToTensor(cosine);
		cppflow::tensor max = cppflow::arg_max(cosine_tensor, 0);
		int maxElementIndex = max.get_data<int64_t>()[0];
		float maxElement = cosine[maxElementIndex];
		currentSubNo = std::get<1>(vector_sub_copy[maxElementIndex]);
		currentSubLenght = std::get<2>(vector_sub_copy[maxElementIndex]);
		nextVector = std::get<3>(vector_sub_copy[maxElementIndex]);
		show = "Subtitle: " + ofToString(currentSubNo) + " - " + ofToString(currentSubNo + currentSubLenght) + "\n\nCosine similarity: " + ofToString(maxElement) + "\n\nSubtitles left: " + ofToString(vector_sub_copy.size() - 1);
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
	std::vector<float> cosine;
	for (auto& element : vector_sub_copy) {
		cppflow::tensor cosine_similarity = cppflow::sum(nextVector * std::get<0>(element), cppflow::tensor({ 1 }));
		cosine.push_back(cosine_similarity.get_data<float>()[0]);
	}
	cppflow::tensor cosine_tensor = ofxTF2::vectorToTensor(cosine);
	cppflow::tensor max = cppflow::arg_max(cosine_tensor, 0);
	int maxElementIndex = max.get_data<int64_t>()[0];
	float maxElement = cosine[maxElementIndex];
	currentSubNo = std::get<1>(vector_sub_copy[maxElementIndex]);
	currentSubLenght = std::get<2>(vector_sub_copy[maxElementIndex]);
	nextVector = std::get<3>(vector_sub_copy[maxElementIndex]);
	show = "Subtitle: " + ofToString(currentSubNo) + " - " + ofToString(currentSubNo + currentSubLenght) + "\n\nCosine similarity: " + ofToString(maxElement) + "\n\nSubtitles left: " + ofToString(vector_sub_copy.size() - 1);
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
std::string ofApp::chatbot(std::string str) {
		int maxElementIndex = 0;
		std::vector<int> input_vector = textEncoder.encode(str);
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
			std::vector<cppflow::tensor> vectorOfOutputTensors = bot.runMultiModel({ input_1, input_2 });
			vectorOfOutputTensors[0] = cppflow::slice(vectorOfOutputTensors[0], cppflow::tensor({ 0, i, 0 }), cppflow::tensor({ 1, 1, -1 }), cppflow::datatype(TF_FLOAT));
			cppflow::tensor max = cppflow::arg_max(vectorOfOutputTensors[0], 2);
			maxElementIndex = max.get_data<int64_t>()[0];
			output_vector.push_back(maxElementIndex);
			max = cppflow::cast(max, TF_INT32, TF_FLOAT);
			input_2 = cppflow::concat(1, { input_2, max });
			input_2 = cppflow::cast(input_2, TF_INT32, TF_FLOAT);
		}

		output_vector.pop_back();
		output_vector.erase(output_vector.begin());
		decoded_answer = textEncoder.decode(output_vector);
		ofStringReplace(decoded_answer, "_", " ");
		decoded_answer = std::regex_replace(decoded_answer, std::regex("\\s+"), " ");
		decoded_answer = std::regex_replace(decoded_answer, std::regex("\\s([+.!?])"), "$1");
		return decoded_answer;
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