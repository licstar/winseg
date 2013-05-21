#include <string>
#include <map>
#include <vector>
using namespace std;

#define MAX_STRING 100

void readWord(char *word, FILE *fin){
	int a=0, ch;

	while (!feof(fin)) {
		ch=fgetc(fin);

		if (ch==13) continue;

		if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
			if (a>0) {
				if (ch=='\n') ungetc(ch, fin);
				break;
			}

			if (ch=='\n') {
				strcpy(word, (char *)"</s>");
				return;
			}
			else continue;
		}

		word[a]=ch;
		a++;

		if (a>=MAX_STRING) {
			printf("Too long word found!\n");   //truncate too long words
			a--;
		}
	}
	word[a]=0;
}

char allwords[150000];

map<string, char*> chkch;
int allwordsLen = 0;
map<string, int> chk;

char *addRealWords(char *word){
	string w = word;
	map<string, char*>::iterator it = chkch.find(w);
	if(it != chkch.end()){
		return it->second;
	}else{
		int len = strlen(word);
		char *ret = allwords+allwordsLen;
		strcpy(allwords+allwordsLen, word);
		allwordsLen += len+1;
		chkch[word] = ret;
		return ret;
	}
}

data_t addWord(char *word){
	data_t ret;
	string w = word;
	map<string, int>::iterator it = chk.find(w);
	if(it != chk.end()){
		ret.word = it->second;
	}else{
		ret.word = chk.size();
		chk[w] = ret.word;
	}
	ret.ch = addRealWords(word);
	return ret;
}

data_t getWord(char *word){
	data_t ret;
	string w = word;
	map<string, int>::iterator it = chk.find(w);
	if(it != chk.end()){
		ret.word = it->second;
	}else{
		ret.word = chk["unknown"];
	}
	ret.ch = addRealWords(word);
	return ret;
}

data_t readWordIndex(FILE *fin, int &tag, bool add=false){
	char word[MAX_STRING];
	data_t ret;
	ret.word = 0;

	readWord(word, fin);
	if (feof(fin)) return ret;

	tag = 0;
	if(strcmp("</s>", word) != 0){
		for(int k = strlen(word)-1; k >=0; k--){
			if(word[k] == '/'){
				tag = atoi(word+k+1);
				word[k] = 0;
				break;
			}
		}
		if(add)
			ret = addWord(word);
		else
			ret = getWord(word);
	}

	return ret;
}

void learnVocab(const char *train_file){
	FILE *fi=fopen(train_file, "rb");
	while(1){
		int tag;
		readWordIndex(fi, tag, true);
		if (feof(fi)) break;
	}
	fclose(fi);
}

void init(const char *train_file){
	//initWithSenna(); //senna的字典
	//tokenizer = initTokenizer(); //分词器
	chk["</s>"] = 0;
	chk["unknown"] = 1;
	chk["padding"] = 2;

	learnVocab(train_file);
//	for(int i = 0; i < 130000; i++){
//		addWord(senna_raw_words[i].str);
//	}
	//readalldata的时候，需要存储别的特征，caps和suff

}


void readAllData(const char *file, const char *dataset, int window_size, data_t *&data, int *&b, int &N, int &uN){
	vector<vector<pair<data_t, int> > > mydata;
	FILE *fi=fopen(file, "rb");
	
	vector<pair<data_t, int> > line;

	data_t padding; //这个想办法初始化一下
	padding.word = 2;
	padding.ch = NULL;

	N = 0;
	while(1){
		int tag;
		data_t dt = readWordIndex(fi, tag);
		if (feof(fi)) break;
		line.push_back(make_pair(dt, tag));

		if(dt.word == 0){
			line.pop_back();
			mydata.push_back(line);
			N += line.size();
			line.clear();
		}
	}
	fclose(fi);

	data = new data_t[N * window_size];
	b = new int[N];

	int hw = (window_size-1)/2;
	//int hw = 2;

	int unknown = 0;
	for(size_t i = 0, offset=0; i < mydata.size(); i++){
		vector<pair<data_t, int> > &vec = mydata[i];

		for(int j = 0; j < (int)vec.size(); j++, offset+=window_size){
			for(int k = hw; k > 0; k--){
				if(j-k >= 0){
					data[offset + hw - k] = vec[j-k].first;
				}else{
					data[offset + hw - k] = padding; //PADDING
				}
			}
			for(int k = 1; k <= hw; k++){
				if(j+k < (int)vec.size()){
					data[offset + hw + k] = vec[j+k].first;
				}else{
					data[offset + hw + k] = padding; //PADDING
				}
			}
			data[offset + hw] = vec[j].first;
			b[offset/window_size] = vec[j].second;
			if(vec[j].first.word == 1)
				unknown++;
		}
	}

	printf("%s data: N(words):%d, unknown:%d\n", dataset, N, unknown);
	uN = unknown;

}