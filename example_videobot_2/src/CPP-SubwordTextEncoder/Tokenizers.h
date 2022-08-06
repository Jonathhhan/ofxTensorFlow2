#include <json.hpp>
#include <string>
#include <list>
#include <map>
#include <vector>
#include <thread>
#include <tuple>
#include <utility>
#include <future>
#include <regex>

using json = nlohmann::json; // For convince sake.
namespace tokenizers {
    class SubwordTextEncoder {
    protected:
        static const std::string END_OF_WORD;
        const std::string HEADER_PREFIX = "### ";
        const std::string METADATA_PREFIX = "### Metadata: ";
        std::map<std::string, int> _subword_to_id = {};
        int _vocab_size;
        std::string _name;
        json _metadata = {};

        // Methods
        std::string _split_word(std::string s);

        std::string _id_to_subword(int id);

        [[nodiscard]] std::list<int> _byte_encode(const std::string &token) const;

        static std::string space_punctuation(const std::string &s);

        static std::list<int> _pad_incr(const std::list<int> &ids);

        static std::list<std::list<std::string>> _chunk_corpus(std::list<std::string> list, unsigned int n);

        const unsigned int PROCESSOR_COUNT = std::thread::hardware_concurrency() * .8;

        static std::list<std::list<std::string>> _batch_word_tokenize(const std::list<std::string> &texts);

        static std::map<std::string, int> _word_counts(const std::list<std::list<std::string>> &tokenized_text);

        static std::list<std::string> _split(const std::string &delimiter, std::string s);


    public:
        unsigned long long int _target_vocab_size;
        std::list<std::string> vocabulary;
        unsigned long long int largest_word;
        unsigned int smallest_word;

        explicit SubwordTextEncoder(unsigned long long int target_vocab_size, std::string name);

        explicit SubwordTextEncoder(const std::string &filename_prefix);

        static std::list<std::string> word_tokenize(const std::string &text);

        [[nodiscard]] int get_vocab_size() const { return _vocab_size; };

        [[nodiscard]] std::list<std::string> get_vocabulary() const { return vocabulary; };
        
        [[nodiscard]] std::string get_name() const {return _name; };

        virtual void build_vocabulary(const std::list<std::string> &texts);

        void write_lines(const std::string &filename_prefix);

        static std::string filename(const std::string &filename_prefix) { return filename_prefix + ".subwords"; };

        std::list<int> encode(const std::string &sentence);

        std::string decode(const std::list<int> &sentence_encoded);
    };
}