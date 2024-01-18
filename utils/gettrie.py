import sys
from trie import Trie

class Gazetteer:
    def __init__(self, lower):
        self.trie = Trie()
        self.ent2type = {}
        self.ent2id = {"<UNK>": 0}
        self.lower = lower
        self.space = ""

    def _process_word_list(self, word_list):
        return [word.lower() for word in word_list] if self.lower else word_list

    def enumerate_match_list(self, word_list):
        processed_word_list = self._process_word_list(word_list)
        return self.trie.enumerate_match(processed_word_list, self.space)

    def insert(self, word_list, source):
        processed_word_list = self._process_word_list(word_list)
        string = self.space.join(processed_word_list)
        self.trie.insert(processed_word_list)
        self.ent2type.setdefault(string, source)
        self.ent2id.setdefault(string, len(self.ent2id))

    def search_id(self, word_list):
        processed_word_list = self._process_word_list(word_list)
        string = self.space.join(processed_word_list)
        return self.ent2id.get(string, self.ent2id["<UNK>"])

    def search_type(self, word_list):
        processed_word_list = self._process_word_list(word_list)
        string = self.space.join(processed_word_list)
        return self.ent2type.get(string, self._handle_type_not_found(string))

    def _handle_type_not_found(self, string):
        print(f"Error in finding entity type at gazetteer.py, exit program! String: {string}")
        exit(0)

    def size(self):
        return len(self.ent2type)
