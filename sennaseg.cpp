#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <omp.h>
#ifdef LINUX
#include <sys/time.h>
#else
#include <time.h>
#endif

using namespace std;

const int H = 100; //隐藏层
const int MAX_C = 50; //最大分类数
const int MAX_F = 1000; //输入层最大的大小
const char *model_name = "model_300_nosuff_noinit";

const char *train_file = "train.txt";
const char *valid_file = "valid.txt";
const char *test_file = "test.txt";
const char *dict_file = "dict.txt";

int class_size; //分类数
int input_size; //特征数，输入层大小 input_size = window_size*vector_size
int window_size; //窗口大小
int vector_size; //一个词单元的向量大小 = 词向量大小（约50） + 所有特征的大小（约10）

//===================== 所有要优化的参数 =====================
struct embedding_t{
	int size; //里面包含多少个变量（value 里面的变量个数） size = element_size * element_num
	int element_size; //一个向量的长度
	int element_num; //向量的个数
	double *value; //所有的参数

	void init(int element_size, int element_num){
		this->element_size = element_size;
		this->element_num = element_num;
		size = element_size * element_num;
		value = new double[size];
	}
};

embedding_t words; //词向量

double *A; //特征矩阵：[分类数][隐藏层] 第二层的权重
double *B; //特征矩阵：[隐藏层][特征数] 第一层的权重
double *gA, *gB;

//===================== 已知数据 =====================
struct data_t{
	int word; //词的编号
	char *ch; //实际的词，用于输出
};
//训练集
data_t *data; //训练数据：[样本数][特征数]
int N; //训练集大小
int uN; //未知词
int *b; //目标矩阵[样本数] 训练集

//验证集
data_t *vdata; //测试数据：[样本数][特征数]
int vN; //测试集大小
int uvN; //未知词
int *vb; //目标矩阵[样本数] 测试集

//测试集
data_t *tdata; //测试数据：[样本数][特征数]
int tN; //测试集大小
int utN; //未知词
int *tb; //目标矩阵[样本数] 测试集


#include "fileutil.hpp"


double time_start;
double lambda = 0;//0.01; //正则项参数权重
double alpha = 0.01; //学习速率
int iter = 0;

const int thread_num = 4;
const int patch_size = thread_num;

double getTime(){
#ifdef LINUX
	timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
#else
	return 0;
#endif
}

double nextDouble(){
	return rand() / (RAND_MAX + 1.0);
}

void softmax(double hoSums[], double result[], int n){
	double max = hoSums[0];
	for (int i = 0; i < n; ++i)
		if (hoSums[i] > max) max = hoSums[i];
	double scale = 0.0;
	for (int i = 0; i < n; ++i)
		scale += exp(hoSums[i] - max);
	for (int i = 0; i < n; ++i)
		result[i] = exp(hoSums[i] - max) / scale;
}

double sigmoid(double x){
	return 1 / (1 + exp(-x));
}

double hardtanh(double x){
	if(x > 1)
		return 1;
	if(x < -1)
		return -1;
	return x;
}

//b = Ax
void fastmult(double *A, double *x, double *b, int xlen, int blen){
	double val1, val2, val3, val4;
	double val5, val6, val7, val8;
	int i;
	for (i=0; i<blen/8*8; i+=8) {
		val1=0;
		val2=0;
		val3=0;
		val4=0;

		val5=0;
		val6=0;
		val7=0;
		val8=0;

		for (int j=0; j<xlen; j++) {
			val1 += x[j] * A[j+(i+0)*xlen];
			val2 += x[j] * A[j+(i+1)*xlen];
			val3 += x[j] * A[j+(i+2)*xlen];
			val4 += x[j] * A[j+(i+3)*xlen];

			val5 += x[j] * A[j+(i+4)*xlen];
			val6 += x[j] * A[j+(i+5)*xlen];
			val7 += x[j] * A[j+(i+6)*xlen];
			val8 += x[j] * A[j+(i+7)*xlen];
		}
		b[i+0] += val1;
		b[i+1] += val2;
		b[i+2] += val3;
		b[i+3] += val4;

		b[i+4] += val5;
		b[i+5] += val6;
		b[i+6] += val7;
		b[i+7] += val8;
	}

	for (; i<blen; i++) {
		for (int j=0; j<xlen; j++) {
			b[i] += x[j] * A[j+i*xlen];
		}
	}
}

double checkCase(data_t *id, int ans, int &correct, int &output, double *p=NULL, bool gd = false){
	double x[MAX_F];
	for(int i = 0, j = 0; i < window_size; i++){
		int offset = id[i].word * words.element_size;
		for(int k = 0; k < words.element_size; k++,j++){
			x[j] = words.value[offset + k];
		}
	}

	double h[H] = {0};
	fastmult(B, x, h, input_size, H);
	for(int i = 0; i < H; i++){
		//h[i] = sigmoid(h[i]);
		h[i] = tanh(h[i]);
		//h[i] = hardtanh(h[i]+biasH[i]);
		//if(h[i] > 1) h[i] = 1;
		//if(h[i] < -1) h[i] = -1;
	}
	//for(int i = 0, k=0; i < H; i++){
	//	for(int j = 0; j < input_size; j++,k++){
	//		//h[i] += x[j] * B[i*input_size+j];
	//		h[i] += x[j] * B[k];
	//	}
	//	h[i] = sigmoid(h[i]);
	//}

	double r[MAX_C] = {0};
	for(int i = 0; i < class_size; i++){
		//r[i] = biasOutput[i];
		for(int j = 0; j < H; j++){
			r[i] += h[j] * A[i*H+j];
		}
	}
	double y[MAX_C];
	softmax(r, y, class_size);

	if(gd){ //修改参数
		/*for(int i = 0; i < class_size; i++){
			if(i == ans){
				biasOutput[i] += alpha*(1-y[i]);
			}else{
				biasOutput[i] += alpha*(0-y[i]);
			}
		}*/

		double dh[H] = {0};
		for(int j = 0; j < H; j++){
			dh[j] = A[ans*H+j];
			for(int i = 0; i < class_size; i++){
				dh[j] -= y[i]*A[i*H+j];
			}
			//dh[j] *= h[j]*(1-h[j]);
			dh[j] *= 1-h[j]*h[j];
			/*if(h[j] > 1 || h[j] < -1)
				dh[j] = 0;
			biasH[j] += alpha * dh[j];*/
		}

		//#pragma omp critical
		{
			for(int i = 0; i < class_size; i++){
				double v = (i==ans?1:0) - y[i];
				for(int j = 0; j < H; j++){
					int t = i*H+j;
					A[t] += alpha/sqrt(H) * (v * h[j] - lambda * A[t]);
					//gA[i*H+j] += v * h[j];
				}
			}

			double dx[MAX_F] = {0};

			//fastmult(B, dh, dx, input_size, H);
			for(int i = 0; i < H; i++){
				for(int j = 0; j < input_size; j++){
					dx[j] += dh[i] * B[i*input_size+j];
				}
			}

			for(int i = 0; i < H; i++){
				for(int j = 0; j < input_size; j++){
					int t = i*input_size+j;
					B[t] += alpha/sqrt(input_size) * (x[j] * dh[i] - lambda * B[t]);
					//gB[i*input_size+j] += -x[j] * dh[i];
				}
			}


			for(int i = 0, j = 0; i < window_size; i++){
				int offset = id[i].word * words.element_size;
				for(int k = 0; k < words.element_size; k++,j++){
					int t = offset + k;
					words.value[t] += alpha * (dx[j] - lambda * words.value[t]);
				}
			}

		}
	}

	output = 0;
	double maxi = 0;
	bool ok = true;
	for(int i = 0; i < class_size; i++){
		if(i != ans && y[i] >= y[ans])
			ok = false;
		if(y[i] > maxi){
			maxi = y[i];
			output = i;
		}
		if(p)
			p[i] = -log(y[i]);
	}

	if(ok)
		correct++;
	return log(y[ans]); //计算似然
}


void writeFile(const char *name, double *A, int size){
	FILE *fout = fopen(name, "wb");
	fwrite(A, sizeof(double), size, fout);
	fclose(fout);
}

//0S 1B 2E 3M
double inf = 1e100;
double viterbi_trans[4][4]={
	{0, 0, inf, inf},
	{inf, inf, 0, 0},
	{0, 0, inf, inf},
	{inf, inf, 0, 0}
};

void viterbi(double dp[][4], int *ret, int len){
	int back[2000][4];
	double rec[2][4];
	double *now = rec[0];
	double *next = rec[1];

	//0S 1B 2E 3M
	now[0] = dp[0][0];
	now[1] = dp[0][1];
	now[2] = inf;
	now[3] = inf;

	for(int i = 1; i < len; i++){
		next[0] = next[1] = next[2] = next[3] = inf;
		for(int j = 0; j < 4; j++){ //上一个节点
			double v = now[j];
			for(int k = 0; k < 4; k++){ //当前节点
				if(next[k] > v + viterbi_trans[j][k] + dp[i][k]){
					next[k] = v + viterbi_trans[j][k] + dp[i][k];
					back[i][k] = j;
				}
			}
		}
		swap(now, next);
	}
	int index = now[0]<now[2]?0:2;
	
	while(len > 0){
		len--;
		ret[len] = index;
		index = back[len][index];
	}
	
}

void checkPRF(int *ans, int *opt, int len, int &cntRight, int &cntAnswerWord, int &cntOutputWord){
	int lans = -1, lopt = -1;

	for (int i = 0; i < len; i++) {
		int v = 0;
		if((ans[i] == 0 || ans[i] == 2) && (opt[i] == 0 || opt[i] == 2)){
			if(lans == lopt)
				cntRight++;
		}

		if(ans[i] == 0 || ans[i] == 2){
			lans = i;
			cntAnswerWord++;
			v++;
		}
		if(opt[i] == 0 || opt[i] == 2){
			lopt = i;
			cntOutputWord++;
			v++;
		}
	}
	if((opt[len-1] != 0 && opt[len-1] != 2) || 
		(ans[len-1] != 0 && ans[len-1] != 2)){
		printf("Error: not end with S/E.\n");
	}
}

double checkSet(const char *dataset, data_t *data, int *b, int N, char *fname = NULL){
	int hw = (window_size-1)/2;

	FILE *fout = NULL;
	FILE *fout2 = NULL;
	if(fname) {
		fout = fopen(fname, "w");
		char fname2[100];
		sprintf(fname2, "%s_test.txt", fname);
		fout2 = fopen(fname2, "w");

	}
	double ret = 0;
	int wordCorrect = 0; //直接的词准确率
	int viterbiCorrect = 0; //viterbi之后的词准确率
	int cntRight = 0;
	int cntAnswerWord = 0;
	int cntOutputWord = 0;


	double dp[2000][4];
	int label[2000];
	int ans[2000];
	char *chs[2000];
	int index = 0;

	for(int s = 0; s < N; s++){
		int tc = 0;
		int output;
		ret += checkCase(data+s*window_size, b[s], tc, output, dp[index]);
		wordCorrect += tc;

		if(fname){
			double mini = 1e300;
			for(int i=0;i<4;i++){
				if(i!=b[s] && dp[index][i]<mini){
					mini=dp[index][i];
				}
			}
			fprintf(fout2, "%lf\t%lf\n", dp[index][b[s]], mini);
		}

		ans[index] = b[s];
		chs[index] = (*(data+s*window_size+hw)).ch;
		index++;

		if((*(data+s*window_size+hw+1)).word == 2){ //下一个是padding
			viterbi(dp, label, index);
			for(int i = 0; i < index; i++){
				if(fname) fprintf(fout, "%s", chs[i]);
				if(label[i] == 0 || label[i] == 2){
					if(fname) fprintf(fout, " ");
				}
				if(ans[i] == label[i])
					viterbiCorrect++;
			}
			if(fname) fprintf(fout, "\n");

			checkPRF(ans, label, index, cntRight, cntAnswerWord, cntOutputWord);
			index = 0;
		}
	}
	if(fname){
		fclose(fout);
		fclose(fout2);
	}
	double P = 1.0 * cntRight / cntOutputWord;
	double R = 1.0 * cntRight / cntAnswerWord;
	double F = 2 * P * R / (P + R);
	printf("%s:%lf(%.2lf%%,%.2lf%%|%.2lf%%,%.2lf%%,%.2lf%%), ", dataset, -ret/N,
		100.*wordCorrect/N, 100.*viterbiCorrect/N, 100.*P, 100.*R, 100.*F);
	return -ret/N;
}

//检查正确率和似然
//返回值是似然
double check(){
	double ret = 0;
	char fname[100];

	double ps = 0;
	int pnum = 0;
	for(int i = 0; i < class_size*H; i++,pnum++){
		ps += A[i]*A[i];
	}
	for(int i = 0; i < H*input_size; i++,pnum++){
		ps += B[i]*B[i];
	}
	for(int i = 0; i < words.size; i++,pnum++){
		ps += words.value[i]*words.value[i];
	}

	sprintf(fname, "%s_A", model_name);
	writeFile(fname, A, class_size*H);
	sprintf(fname, "%s_B", model_name);
	writeFile(fname, B, H*input_size);
	sprintf(fname, "%s_w", model_name);
	writeFile(fname, words.value, words.size);


	printf("para: %lf, ", ps/pnum/2);

	ret = checkSet("train", data, b, N);
	checkSet("valid", vdata, vb, vN);
	sprintf(fname, "%s_%d_output", model_name, iter);
	checkSet("test", tdata, tb, tN, fname);

	printf("time:%.1lf\n", getTime()-time_start);
	fflush(stdout);

	double fret = ret + ps/pnum*lambda/2;
	return fret;
}

int readFile(const char *name, double *A, int size){
	FILE *fin = fopen(name, "rb");
	if(!fin)
		return 0;
	int len = (int)fread(A, sizeof(double), size, fin);
	fclose(fin);
	return len;
}

//把训练集中没有出现过的字的词向量，全部替换为unknown
vector<bool> wordsExists;
void checkWordsExists(){
	wordsExists = vector<bool>(words.element_num);
	for(int i=0, k=0; i < N; i++){
		for(int j = 0; j < window_size; j++,k++){
			wordsExists[data[k].word] = true;
		}
	}
	int cnt = 0;
	for(int i = 0; i < words.element_num; i++){
		if(wordsExists[i]){
			cnt++;
		}
	}
	printf("words exists:%d\n", cnt);
}
void updateWordsExists(){
	for(int i = 0; i < words.element_num; i++){
		if(!wordsExists[i]){
			for(int j = 0; j < words.element_size; j++){
				words.value[i*words.element_size+j] = words.value[1*words.element_size+j]; //把词替换成unknown
			}
		}
	}
}

int main(int argc, char **argv){
	if(argc < 3){
		printf("Useage: ./seg w(null) model_name\n");
		return 0;
	}
	model_name = argv[2];

	printf("read data size\n");

	window_size = 5;
	class_size = 4;

	vector_size = 50;

	init(dict_file);

	words.init(vector_size, chk.size());

	printf("read data\n");
	readAllData(train_file, "Train", window_size, data, b, N, uN);
	readAllData(valid_file, "Valid", window_size, vdata, vb, vN, uvN);
	readAllData(test_file, "Test", window_size, tdata, tb, tN, utN);

	input_size = window_size * vector_size;

	printf("init. input(features):%d, hidden:%d, output(classes):%d, alpha:%lf, lambda:%.16lf\n", input_size, H, class_size, alpha, lambda);
	printf("window_size:%d, vector_size:%d, vocab_size:%d, allwordsLen:%d, lineMax:%d\n", window_size, vector_size, words.element_num, allwordsLen, lineMax);
	//checkWordsExists();

	A = new double[class_size*H];
	gA = new double[class_size*H];
	B = new double[H*input_size];
	gB = new double[H*input_size];

	for(int i = 0; i < class_size * H; i++){
		A[i] = (nextDouble()-0.5) / sqrt(H);
	}
	for(int i = 0; i < H * input_size; i++){
		B[i] = (nextDouble()-0.5) /sqrt(input_size);
	}
	for(int i = 0; i < words.size; i++){
		words.value[i] = (nextDouble()-0.5);
	}
	
	
	if(readFile(argv[1], words.value, words.size)){
		printf("initialized with %s\n", argv[1]);
		double sum = 0;
		for(int i = 0; i < words.size; i++){
			sum += words.value[i]*words.value[i];
		}
		sum = sqrt(sum/words.size*12);
		for(int i = 0; i < words.size; i++){
			words.value[i] /= sum;
		}
		if(argc > 3){
			double v = atof(argv[3]);
			printf("x%lf %s\n", v);
			for(int i = 0; i < words.size; i++){
				words.value[i] *= v;
			}
		}
	}else{
		printf("not initialized\n");
	}

	for(int i = 0; i < words.element_num; i++){
		for(int j = 0; j < words.element_size; j++){
			//words.value[i * words.element_size + j] = senna_raw_words[i].vec[j] / sqrt(12);
		}
	}
	

	time_start = getTime();

	int *order = new int[N];
	for(int i = 0; i < N; i++){
		order[i] = i;
	}

	//double lastLH = 1e100;
	while(1){
		//计算正确率
		printf("%citer: %d, ", 13, iter);
		//double LH = check();
		//updateWordsExists();
		check();
		iter++;
		/*if(LH > lastLH){
			alpha = 0.0001;
		}
		lastLH = LH;*/


		double lastTime = getTime();
		//memset(gA, 0, sizeof(double)*class_size*H);
		//memset(gB, 0, sizeof(double)*H*input_size);

		for(int i = 0; i < N; i++){
			swap(order[i], order[rand()%N]);
		}
		double tlambda = lambda;
		for(int i = 0; i < N; i++){
			lambda = 0;
			if(i % 10 == 0)
				lambda = tlambda;
			int s = order[i];
			data_t *x = data+s*window_size;
			int ans = b[s];

			int tmp, output;
			checkCase(x, ans, tmp, output, NULL, true);

			if ((i%1000)==0){
				printf("%cIter: %3d\t   Progress: %.2f%%   Words/sec: %.1f ", 13, iter, 100.*i/N, i/(getTime()-lastTime));
			}
		}
		lambda = tlambda;
		//for(int i = 0; i < vN; i++){
		//	int s = i;
		//	data_t *x = vdata + s * window_size;
		//	int ans = vb[s];
		//	int tmp;
		//	checkCase(x, ans, tmp, true);

		//	if ((i%100)==0){
		//	//	printf("%cIter: %3d\t   Progress: %.2f%%   Words/sec: %.1f ", 13, iter, 100.*i/N, i/(getTime()-lastTime));
		//	}
		//}
		//printf("%c", 13);
	}
	return 0;
}