#include <iostream>

using namespace std;

void printVector(vector<int>* v){
	for (int i = 0; i < v->size(); i++){
		cout << "i,";
	}
	cout << endl;
}

class ket {
	public:
		double coeff;
		vector<int> quanta;
		vector<int> indices;

		ket(double c, vector<int>* q){
			coeff = c;
			quanta = *q;
			vector<int> placeholder;
			for (int i = 0; i < q->size(); i++){
				placeholder.push_back(i);
			}
			indices = placeholder;
		}

		void listValues(){
			cout << "Coeff: " << this.coeff << endl;
			cout << "Quanta: " << endl;
			printVector(&(this.quanta));
			cout << "Indices: " << endl;
			printVector(&(this.indices));
		}
};

int main(int argc, char *argv[]){
	
}
