#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

using namespace std;
using namespace Spectra;

//Set some global variables. These are just the default values and will probaby be modified later.
int reach = 4;
int budget = 2;

double m0 = 0.3; //This makes sure the correlation length, 1/m0, is a few lattice sites
double mbar = 4.0/3.14159;

double cutoff = pow(10,-11); //If the coefficient of a ket is below this cutoff, assume it's just a round-off error from 0 and delete it.
bool initialBasisRestrictions = false; //Cut off certain computations early if I only want to deal with states I'd generate in the original basis, determined by the reach and budget

void printVector(vector<int>* v){
	for (int i = 0; i < v->size(); i++){
		cout << v->at(i) << ",";
	}
	cout << endl;
}

class ket {
	public:
		double coeff;
		vector<int> quanta;
		vector<int> indices;
		int basisTag;

		ket(double c, vector<int>* q){
			coeff = c;
			quanta = *q;
			vector<int> placeholder;
			for (int i = 0; i < q->size(); i++){
				placeholder.push_back(i);
			}
			indices = placeholder;
			basisTag = -1; //this is a default, meaning it hasn't been tagged as part of a basis.
		}

		void listValues(){
			cout << "Coeff: " << coeff << endl;
			cout << "Quanta: " << endl;
			printVector(&(quanta));
			cout << "Indices: " << endl;
			printVector(&(indices));
		}

		int totalQuanta(){
			int runningTotal = 0;
			for (int i = 0; i < quanta.size(); i++){
				runningTotal += quanta.at(i);
			}
			return runningTotal;
		}
};

void listStateValues(vector<ket>* state){
	for (int i = 0; i < state->size(); i++){
		(state->at(i)).listValues();
		cout << endl;
	}
}

//Given a seed state, it adds quanta from the budget to all possible sites recursively
void allocateQuanta(vector<vector<int>>* finishedStates, vector<int> seedState, int leftoverBudget){

	//First, if the seed state is already as long as I'm allowing them to get, or if I'm out of budget, finish
	if ((seedState.size() == reach) || (leftoverBudget == 0)){
		finishedStates->push_back(seedState);
		return;
	}

	//Cycle over all the ways I can spend the budget on the next site. Then, call the function again to allocate on the next site.
	for (int i = 0; i <= leftoverBudget; i++){
		vector<int> tempState = seedState;
		tempState.push_back(i);
		allocateQuanta(finishedStates, tempState, leftoverBudget - i);
	}
}

//Kick off the recursion
void beginAllocatingQuanta(vector<vector<int>>* finishedStates){

	//Throw in the ground state too, because it won't be covered in my algorithm
	vector<int> trialGroundQuanta = {0};
	finishedStates->push_back(trialGroundQuanta);

	//First, spend the whole budget on the first site.
	vector<int> maxQuanta = {budget};
	finishedStates->push_back(maxQuanta);

	//Cycle through spending some nonzero part of the budget on the first site (has to be nonzero for uniqueness purposes)
	for (int i = 1; i < budget; i++){
		vector<int> seedState = {i};
		allocateQuanta(finishedStates,seedState,budget-i);
	}
}

//Implements a custom ranking on the kets. 0 means the first ket is greater, and 1 means the second ket is greater.
int rankTwoKets(ket* k0, ket* k1){
	vector<int> q0 = k0->quanta;
	vector<int> q1 = k1->quanta;

	int minSize = q0.size();
	if (q1.size() < minSize){
		minSize = q1.size();
	}

	//Rank them in alphabetical order
	for (int i = 0; i < minSize; i++){
		if (q0.at(i) < q1.at(i)){
			return 1;
		}
		if (q0.at(i) > q1.at(i)){
			return 0;
		}
	}

	//If they're all equal so far, say the longer one is greater
	if (q0.size() > q1.size()){
		return 0;
	}
	if (q0.size() < q1.size()){
		return 1;
	}

	//If they've passed all test, they're the same ket. Return -1
	return -1;
}

//Implements a quicksort on a list of kets using the above ordering
vector<ket> sortKetList(vector<ket> state){
	int rankResult;
	
	//Handle this differently if there are only 0, 1, 2 kets
	if (state.size() < 2){
		return state;
	}
	if (state.size() == 2){
		rankResult = rankTwoKets(&(state.at(0)), &(state.at(1)));
		if (rankResult == 0){ //Flip the ordering
			vector<ket> tempState;
			tempState.push_back(state.at(1));
			tempState.push_back(state.at(0));
			return tempState;
		} else{
			return state;
		}
	}

	int pivot = state.size() / 2;
	ket pivotKet = state.at(pivot);
	vector<ket> lesserList;
	vector<ket> greaterList;
	for (int i = 0; i < state.size(); i++){
		if (i == pivot){
			continue; //We already know how to handle that
		}
		rankResult = rankTwoKets(&pivotKet, &(state.at(i)));
		if (rankResult == 0){
			lesserList.push_back(state.at(i));
		}
		else{
			greaterList.push_back(state.at(i));
		}
	}
	//Now that they've been sorted around the pivot ket, recurse and sort each sub-list
	lesserList = sortKetList(lesserList);
	greaterList = sortKetList(greaterList);
	//Now concatenate the lists together, least to greatest
	lesserList.push_back(pivotKet);
	lesserList.insert(lesserList.end(), greaterList.begin(), greaterList.end());

	return lesserList;
}

//Does the same quicksort, but sorts them by largest to smallest coeffs instead of the quanta dictionary ranking
vector<ket> sortKetsByCoeff(vector<ket> state){
        //Handle this differently if there are only 0, 1, 2 kets
        if (state.size() < 2){
                return state;
        }
        if (state.size() == 2){
                if (abs(state.at(1).coeff) > abs(state.at(0).coeff)){ //Flip the ordering
                        vector<ket> tempState;
                        tempState.push_back(state.at(1));
                        tempState.push_back(state.at(0));
                        return tempState;
                } else{
                        return state;
                }
        }

        int pivot = state.size() / 2;
        double pivotCoeff = state.at(pivot).coeff;
        vector<ket> lesserList;
        vector<ket> greaterList;
        for (int i = 0; i < state.size(); i++){
                if (i == pivot){
                        continue; //We already know how to handle that
                }
                if (abs(state.at(i).coeff) < abs(pivotCoeff)){
                        lesserList.push_back(state.at(i));
                }
                else{
                        greaterList.push_back(state.at(i));
                }
        }
        //Now that they've been sorted around the pivot ket, recurse and sort each sub-list
        lesserList = sortKetsByCoeff(lesserList);
        greaterList = sortKetsByCoeff(greaterList);
        //Now concatenate the lists together, least to greatest
        greaterList.push_back(state.at(pivot));
        greaterList.insert(greaterList.end(), lesserList.begin(), lesserList.end());

        return greaterList;

}

//Checks if a ket is the trial ground state with no excitations.
bool isTrialGroundState(ket* k){
	
	vector<int> localQuanta = k->quanta;
	for (int i = 0; i < localQuanta.size(); i++){
		if (localQuanta.at(i) != 0){
			return false; //Return false if any of the quanta are nonzero.
		}
	}
	//If it made it through the whole loop and found nothing nonzero, return true.
	return true;
}

//Get rid of leading or trailing zeroes.
void trimZeroes(vector<ket>* states){

	for (int i = 0; i < states->size(); i++){

		if (isTrialGroundState(&(states->at(i)))){ //Handle the trial ground state separately
			(states->at(i)).quanta = {0};
			(states->at(i)).indices = {0};
			continue;
		}

		//Shave zeroes off the end. It'll be more computationally efficient to do this first.
		while ( (states->at(i)).quanta.at( (states->at(i)).quanta.size() - 1) == 0){
			(states->at(i)).quanta.pop_back();
			(states->at(i)).indices.pop_back();
		}

		//Shave zeroes off the beginning.
		while ( (states->at(i)).quanta.at(0) == 0){
			(states->at(i)).quanta.erase( (states->at(i)).quanta.begin() );
			(states->at(i)).indices.erase( (states->at(i)).indices.begin() );
		}
	}

}

void makeBasis(vector<vector<ket>>* useBasis, vector<ket>* opBasis, vector<ket>* matrixBasis, int maxQuanta, int length){

        //Set the budget and reach
        budget = maxQuanta;
        reach = length;

        //Get all the quanta distributions
        vector<vector<int>> quantaDistributions;
        beginAllocatingQuanta(&quantaDistributions);

        //Make kets based on these
        for (int i = 0; i < quantaDistributions.size(); i++){
                ket k(1,&(quantaDistributions.at(i)));
                opBasis->push_back(k);
        }

        //Sort the new bases and add basis tags. Copy it over to the other variables too.
        *opBasis = sortKetList(*opBasis);
        trimZeroes(opBasis);
        for (int i = 0; i < opBasis->size(); i++){
                (opBasis->at(i)).basisTag = i;
                vector<ket> tempState;
                tempState.push_back(opBasis->at(i));
                useBasis->push_back(tempState);
        }
        *matrixBasis = *opBasis;
}

//Checks if two kets have the same quanta
bool sameQuanta(ket* k1, ket* k2){


	//Trim zeroes off both before I continue.
	vector<ket> state1;
	state1.push_back(*k1);
	state1.push_back(*k2);
	trimZeroes(&state1);

	//If their quanta aren't the same size after trimming, we know they're different
	if ( (k1->quanta).size() != (k2->quanta).size() ){
		return false;
	}

	//Now go through the quanta at each location
	for (int i = 0; i < (k1->quanta).size(); i++){
		if ((k1->quanta).at(i) != (k2->quanta.at(i)) ){
			return false;
		}
	}

	return true; //Do this if it gets through every other test.
}

//Take the inner product of two lists of kets. Sorts them first so it's faster to know when a resulting ket isn't in the list and I should give up. Otherwise, it becomes and O(n^2) problem.
double innerProduct(vector<ket>* stateL, vector<ket>* stateR){

	vector<ket> stateRresult = sortKetList(*stateR);
	vector<ket> stateLresult = sortKetList(*stateL);

	vector<int> tempQ = {0};
	ket tempK(1,&tempQ);
	int compResult;
	double total = 0;
	for (int i = 0; i < stateRresult.size(); i++){
		tempK = stateRresult.at(i);
		for (int j = 0; j < stateLresult.size(); j++){
			compResult = rankTwoKets(&(stateLresult.at(j)), &tempK);
			if (compResult == -1){//These are the same ket. Multiply their coefficients, add them to the total, delete the ket from the L list, and move on
				total += (stateLresult.at(j).coeff)*(tempK.coeff);
				stateLresult.erase(stateLresult.begin() + j);
			} else if (compResult == 0){//I couldn't find the ket in the L list. Move on
				break;
			}
		}
	}

	return total;

}

//Perform a binary search in the sorted ket list opBasis and then returns the basis tag of the matched ket
int findKetIndex(ket* k, vector<ket>* opBasis){

	//Copy the binary search algorithm from Wikipedia
	int L = 0;
	int R = opBasis->size() - 1;
	int m;
	int compResult;

	while (L <= R){
		m = (L+R)/2;
		compResult = rankTwoKets(k, &(opBasis->at(m)));
		if (compResult == 0){ //This means k is greater than the ket at m
			L = m+1;
		}
		if (compResult == 1){ //This means k is less than the ket at m
			R = m-1;
		}
		if (compResult == -1){ //This means we have a perfect match
			return (opBasis->at(m)).basisTag;
		}
	}

	//If it made it through the search without finding a match, return -1 to say the ket isn't in the basis
	return -1;

}

//Given a position where a raising or lowering operator would like to act, it pads out the quanta with the necessary zeroes and returns the index of where the operator should act.
int adjustKetIndices(ket* k, int index){

	vector<int> localQuanta = k->quanta;
	vector<int> localIndices = k->indices;

	//First, if the operator acts before the ket starts, pad out the front with zeroes
	int prevLowestIndex;
	while (index < localIndices.at(0)){
		localQuanta.insert(localQuanta.begin(),0);
		prevLowestIndex = localIndices.at(0);
		localIndices.insert(localIndices.begin(),prevLowestIndex - 1 );
	}

	//Now, run through the whole list and see where the desired acting index matches the ket's index
	for (int i = 0; i < localIndices.size(); i++){
		if (index == localIndices.at(i)){
			k->quanta = localQuanta; //Save the data in the original ket
			k->indices = localIndices;
			return i; //Once it finds a match, return the vector-index of the site it should act on
		}
	}

	//If it hasn't found a match, that means I need to pad out zeroes at the end
	int prevHighestIndex;
	while (index != localIndices.at(localIndices.size() - 1)){
		localQuanta.push_back(0);
		prevHighestIndex = localIndices.at(localIndices.size() - 1);
		localIndices.push_back(prevHighestIndex + 1);
	}

	//Save the data in the original ket
	k->quanta = localQuanta;
	k->indices = localIndices;
	return localIndices.at(localIndices.size() - 1);

}

//Get rid of any kets whose coefficients are zero
void prune(vector<ket>* state){

	int counter = 0; //Do this as a while loop because the size of the list is variable
	while (counter < state->size()){
		if (abs((state->at(counter)).coeff) < cutoff){
			state->erase(state->begin() + counter);//It's within a round-off error of zero, so erase it.
		} else{
			counter++; //This state is good; move on.
		}
	}
}

//In a single vector of kets, combine like terms.
void mergeKets(vector<ket>* state){

	//First, sort the list, so you only have to check kets right next to each other
	*state = sortKetList(*state);

	//Clean up the kets by trimming zeroes so they can be properly compared.
	trimZeroes(state);

	int counter = 0;
	while (counter < ((int) state->size() - 1)){
		if (abs(state->at(counter).coeff) < cutoff){
			state->erase(state->begin() + counter);
			continue;
		}//First, if the coeff is 0, delete it and move on.

		if (sameQuanta( &(state->at(counter)), &(state->at(counter+1)) )){
			(state->at(counter)).coeff = (state->at(counter)).coeff + (state->at(counter+1)).coeff;//These kets are the same, so combine their coeffs.
			state->erase(state->begin() + counter + 1); //Delete the ket we combined
		} else{
			counter++; //We know this ket doesn't match the next one on the sorted list, so move on.
		}
	}

	//Finally, clean out any kets that had coefficients of zero after merging them.
	prune(state);
}

//Merge two vectors of kets into one, combining like terms.
void mergeKetLists(vector<ket>* state1, vector<ket>* state2){

	state1->insert(state1->end(), state2->begin(), state2->end());
	mergeKets(state1);

}

//Multiply all the kets in a linear combination by a coefficient
void multiplyOverKets(vector<ket>* state, double c){

	for (int i = 0; i < state->size(); i++){
		(state->at(i)).coeff = (state->at(i)).coeff * c;
	}
}

//Apply the a lowering operator to a linear combination of kets at index x
void aOp(vector<ket>* state, int x){
	int tempQuanta;
	int actIndex;
	ket* k;

	//Apply it to each ket in the vector individually
	for (int i = 0; i < state->size(); i++){
		k = &(state->at(i)); //Make a local pointer of the ket
		//First, check if the index is too low or too high and will just act on a 0
		if (x < k->indices.at(0) || x > k->indices.at(k->indices.size() - 1) ){
			k->coeff = 0;
			continue; //Mark it for deletion and move on
		}
		//Otherwise, get the act index, decrement that quanta, adjust the coefficient, and move on
		actIndex = adjustKetIndices(k, x);
		tempQuanta = k->quanta.at(actIndex);
		k->coeff = k->coeff * sqrt(tempQuanta);
		k->quanta.at(actIndex) = tempQuanta - 1;
	}

	//Now weed out the states that got annihilated.
	prune(state);

	//I might have made some extra zeroes hanging on, so clean those off
	trimZeroes(state);
}

//Apply the raising operator to a linear combination of kets at index x
void aDagOp(vector<ket>* state, int x){
	int tempQuanta;
	int actIndex;
	int lastIndex;
	int firstIndex;
	ket* k;

	//Apply it to each ket in the vector individually
	for (int i = 0; i < state->size(); i++){
		k = &(state->at(i));
		if (initialBasisRestrictions){ //If we're only going to be comparing this against kets in the original basis, see if it breaks the budget or reach requirements
			if (k->totalQuanta() >= budget){
				k->coeff = 0; //Adding one more quanta will make it exceed the budget constraint, so mark it for deletion
				continue;
			}
			firstIndex = (k->indices).at(0);
			lastIndex = (k->indices).at((k->indices).size() - 1);
			if (x < firstIndex){ //We could make it too long by adding indices before...
				if ((lastIndex - x + 1) > reach){
					k->coeff = 0;
					continue;
				}
			}
			if (x > lastIndex){ //...or by adding indices after
				if ((x - firstIndex + 1) > reach){
					k->coeff = 0;
					continue;
				}
			}
		
		}
		//Otherwise, get the act index, decrement that quanta, adjust the coefficient, and move on
		actIndex = adjustKetIndices(k, x);
		tempQuanta = k->quanta.at(actIndex);
		k->coeff = k->coeff * sqrt(tempQuanta + 1);
		k->quanta.at(actIndex) = tempQuanta + 1;
	}

	//Now weed out the states that were marked for deletion by being out of the basis
	if (initialBasisRestrictions){
		prune(state);
	}
}

//The normal-ordered phi^2 operator at some point x
vector<ket> phiSquaredNO(vector<ket>* state, int x){

	vector<ket> term1 = *state;
	vector<ket> term2 = *state;
	vector<ket> term3 = *state;

	aOp(&term1, x);
	aOp(&term1, x);

	aDagOp(&term2, x);
	aDagOp(&term2, x);

	aOp(&term3, x);
	aDagOp(&term3, x);
	multiplyOverKets(&term3, 2);

	mergeKetLists(&term2, &term3);
	mergeKetLists(&term1, &term2);

	multiplyOverKets(&term1, 0.5/mbar);

	return term1;
}

//The cross-term operator phi_x phi_x+a
vector<ket> phixphixpa(vector<ket>* state, int x, int a){

	vector<ket> term1 = *state;
	vector<ket> term2 = *state;
	vector<ket> term3 = *state;
	vector<ket> term4 = *state;

	aOp(&term1, x);
	aOp(&term1, x+a);

	aDagOp(&term2, x);
	aDagOp(&term2, x+a);

	//For terms like these, always lead with the lowering operator. Starting with raising might push it out of budget, and it gets cut prematurely.
	aOp(&term3, x);
	aDagOp(&term3, x+a);

	aOp(&term4, x+a);
	aDagOp(&term4, x);

	mergeKetLists(&term3, &term4);
	mergeKetLists(&term2, &term3);
	mergeKetLists(&term1, &term2);

	multiplyOverKets(&term1, 0.5 / mbar);

	return term1;
}

//This is the translationally-invariant sum of the phi_x phi_x+a operator. I make it act on one ket at a time since the trial ground state has to be handled differently... but I could probably make this run faster.
vector<ket> correlator(vector<ket>* state, int a){

	vector<ket> result;
	vector<ket> tempResult;
	int ketSize;

	for (int i = 0; i < state->size(); i++){
		vector<ket> preppedState;
		preppedState.push_back(state->at(i));
		if (isTrialGroundState(&(state->at(i)))){ //Don't sum over the trial ground state; it has to be handled differently
			tempResult = phixphixpa(&preppedState,0,a);
			mergeKetLists(&result, &tempResult);
			continue;
		}
		ketSize = (state->at(i)).quanta.size();
		for (int x = -(reach - ketSize); x < (reach - 1); x++){
			tempResult = phixphixpa(&preppedState,x,a);
			mergeKetLists(&result, &tempResult);
		}
	}

	return result;
}

//Just the base QHO Hamiltonian, with the constant term dropped, acting on site x
vector<ket> H0op(vector<ket> state, int x){

	aOp(&state, x);
	aDagOp(&state, x);
	multiplyOverKets(&state, mbar);

	return state;
}

//The potential terms for the Hamiltonian, 1/2 (m^2 - mbar^2)phi_x^2 + 1/2 (phi_x+1 - phi_x)^2
vector<ket> Vop(vector<ket> state, int x){

	vector<ket> term1 = state;
	vector<ket> term2 = state;
	vector<ket> term3 = state;

	term1 = phiSquaredNO(&term1, x);
	multiplyOverKets(&term1, 0.5*(m0*m0 - mbar*mbar + 1));

	term2 = phiSquaredNO(&term2, x+1);
	multiplyOverKets(&term2, 0.5);

	term3 = phixphixpa(&term3, x, 1);
	multiplyOverKets(&term3, -1);

	mergeKetLists(&term2, &term3);
	mergeKetLists(&term1, &term2);

	return term1;
}

//The complete Hamiltonian density operator
vector<ket> Hop(vector<ket> state, int x){

	vector<ket> term1 = state;
	vector<ket> term2 = state;

	term1 = H0op(term1, x);
	term2 = Vop(term2, x);

	mergeKetLists(&term1, &term2);

	return term1;
}

vector<ket> HSumOverLSites(vector<ket> state, int start, int finish){

	vector<ket> oneTermResult;
	vector<ket> allResults;

	for (int i = start; i < finish; i++){
		oneTermResult = Hop(state, i);
		mergeKetLists(&allResults, &oneTermResult);
	}

	return allResults;

}

//Creates the sparse matrix for the whole Hamiltonian density
void makeHtrunc(Eigen::SparseMatrix<double>* Htrunc, vector<vector<ket>>* useBasis, vector<ket>* opBasis){

	vector<ket> preOpState;
	vector<ket> postOpState;
	int row;
	int column;
	int startTransInvCounter;
	//Sum over all possible ways the translationally invariant states can be acted on by the Hamiltonian density.
	//But I accomplish this by holding the states still and moving the Hamiltonian density
	
	//Handle the trial ground state differently since it's already translationally invariant.
	if (isTrialGroundState(&(useBasis->at(0).at(0)))){
		startTransInvCounter = 1; //Start the next cycle from 1, since I already took care of 0
		preOpState = useBasis->at(0);
		column = preOpState.at(0).basisTag;
		postOpState = Hop(preOpState, 0);
		for (int j = 0; j < postOpState.size(); j++){
			row = findKetIndex(&(postOpState.at(j)), opBasis);
			if (row != -1){
				Htrunc->coeffRef(row,column) += postOpState.at(j).coeff;
			}
		}
	} else{
		startTransInvCounter = 0;
	}

	for (int i = startTransInvCounter; i < useBasis->size(); i++){
		preOpState = useBasis->at(i);
		if (preOpState.size() > 1){
			cout << "Warning: there are multiple kets in this basis state " << i << endl;
		}
		column = preOpState.at(0).basisTag;
		postOpState = HSumOverLSites(preOpState, -reach, reach);
		for (int j = 0; j < postOpState.size(); j++){
			row = findKetIndex(&(postOpState.at(j)), opBasis);
			if (row != -1){ //If row == -1, the ket is out of the basis
				Htrunc->coeffRef(row, column) += postOpState.at(j).coeff;
			}
		}
	}
}

//Gets the n lowest many eigenvalues and eigenvectors from a sparse matrix
void getGroundStates(Eigen::SparseMatrix<double>* Htrunc, int numValues, vector<ket>* matrixBasis, vector<vector<ket>>* groundStates, vector<double>* groundEnergies){
	//Follow the tutorial from Spectra
	SparseSymMatProd<double> op(*Htrunc);
	SymEigsSolver<SparseSymMatProd<double>> eigs(op, numValues, 2*numValues);
	eigs.init();
	int nconv = eigs.compute(SortRule::SmallestAlge);
	Eigen::VectorXd evalues;
	Eigen::MatrixXd evectors;
	if (eigs.info() == CompInfo::Successful){
		evalues = eigs.eigenvalues();
		evectors = eigs.eigenvectors();
	} else{
		cout << "Warning: the eigenstuff was NOT computed successfully." << endl;
	}

	for (int i = 0; i < numValues; i++){
		groundEnergies->push_back(evalues(i));
	} //Note that this means the lowest eigenvalue, and the ground state, are at the BACK of the vector

	//Now turn the coefficients of the eigenvectors into ket lists
	for (int i = 0; i < numValues; i++){
		vector<ket> eigenstate;
		for (int j = 0; j < matrixBasis->size(); j++){
			ket k = matrixBasis->at(j);
			k.coeff = evectors(j, i);
			if (abs(k.coeff) > cutoff){
				eigenstate.push_back(k);
			}
		}
		eigenstate = sortKetsByCoeff(eigenstate);
		groundStates->push_back(eigenstate);
	}
}

void makeHtruncConverge(double threshold, int numValues, int maxLength, string outputFile){

	//Open the output file
	ofstream energiesOutput;
	energiesOutput.open(outputFile + ".txt");

	//Start with a budget of two and increment by twos after that
	budget = 2;
	reach = maxLength;

	vector<vector<ket>> useBasis;
	vector<ket> opBasis;
	vector<ket> matrixBasis;
	makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach);
	int basisSize = opBasis.size();

	Eigen::SparseMatrix<double> Htrunc(basisSize, basisSize);
        makeHtrunc(&Htrunc, &useBasis, &opBasis);

        vector<vector<ket>> groundStates;
        vector<double> groundEnergies;
        getGroundStates(&Htrunc, numValues, &matrixBasis, &groundStates, &groundEnergies);
	double prevGroundEnergy = groundEnergies.at(groundEnergies.size() - 1);
	cout << "Basis size: " << basisSize << endl;
	cout << "Ground Energy: " << prevGroundEnergy << endl;
	energiesOutput << basisSize << endl;
	energiesOutput << prevGroundEnergy << endl;
	double nextGroundEnergy;

	//Now that I have an initial value, I can start incrementing the budget and compare for convergence
	bool converged;
	while (!converged){
		budget += 2; //Increment the budget. Don't increment reach; that takes forever to converge
		useBasis.clear();
		opBasis.clear();
		matrixBasis.clear();
		makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach); //Make the new basis
		basisSize = opBasis.size();
		cout << "Made a basis of size " << basisSize << " with reach " << reach << " and budget " << budget << endl;

		Htrunc.setZero();
		Htrunc.resize(basisSize, basisSize);
		makeHtrunc(&Htrunc, &useBasis, &opBasis); //Make Htrunc again
		cout << "Made the truncated Hamiltonian." << endl;

		groundStates.clear();
		groundEnergies.clear();
		getGroundStates(&Htrunc, numValues, &matrixBasis, &groundStates, &groundEnergies);
		nextGroundEnergy = groundEnergies.at(groundEnergies.size() - 1);

		cout << "Basis size: " << basisSize << endl;
		cout << "Ground Energy: " << nextGroundEnergy << endl;
		energiesOutput << basisSize << endl;
		energiesOutput << prevGroundEnergy << endl;

		if (abs( (nextGroundEnergy - prevGroundEnergy) / nextGroundEnergy) < threshold){ //If they've converged, finish the process
			converged = true;
			//Now write the kets in the ground state to a file for later use
			string outputFileGS = outputFile + "GroundStates.txt";
			ofstream statesOutput;
			statesOutput.open(outputFileGS);
			//Immediately put in the key variables so those can be loaded later.
			statesOutput << "m0=" << m0 << endl;
			statesOutput << "mbar=" << mbar << endl;
			statesOutput << "reach=" << reach << endl;
			vector<ket> state;
			vector<int> localQuanta;
			double normTotal;
			for (int i = 0; i < groundStates.size(); i++){
				state = groundStates.at(groundStates.size() - 1 - i);
				statesOutput << i << "," << groundEnergies.at(groundStates.size() - 1 - i) << endl;
				normTotal = 0;
				int ketCounter = 0;
				while (normTotal < 0.99){ //Write states until I've covered 99% of the state's norm
					statesOutput << (state.at(ketCounter)).coeff << endl;
					normTotal += (state.at(ketCounter)).coeff * (state.at(ketCounter)).coeff;
					localQuanta = (state.at(ketCounter)).quanta;
					for (int k = 0; k < localQuanta.size(); k++){
						statesOutput << localQuanta.at(k) << ",";
					}
					statesOutput << endl;
					ketCounter++;
				}
				statesOutput << "x" << endl; //This is a signal for the ket reader functions to move on
			}
			statesOutput.close();
		} else{
			prevGroundEnergy = nextGroundEnergy;
		}
	}

	energiesOutput.close();

}

void loadGroundStateFromFile(vector<ket>* groundStateKets, string filename){
	ifstream inputFile;
	inputFile.open(filename);

	string oneLine;
	//Load in key information about the global variables used to make that eigenstate
	inputFile >> oneLine;
	cout << oneLine << endl;
	m0 = stod(oneLine.substr(3));
	inputFile >> oneLine;
	mbar = stod(oneLine.substr(5));
	inputFile >> oneLine;
	reach = stod(oneLine.substr(6));

	double coeff;
	vector<int> q;
	int i;
	inputFile >> oneLine; //Get through the first line, which is just an int that labels which eigenstate it is
	bool keepGoing = true;
	int emergencyBreak = 0;
	while (keepGoing){
		emergencyBreak++;
		if (emergencyBreak > 1000){
		//	break;
		}

		inputFile >> oneLine;
		if (oneLine.find('x') != -1){
			break; //This means it found the end of the eigenstate in the file and is moving o
		}
		coeff = stod(oneLine); //Load the coefficient
		inputFile >> oneLine; //Load the next line, which is the quanta
		stringstream quantaStream(oneLine);
		string oneQuanta;
		while (quantaStream.good()){
			getline(quantaStream, oneQuanta, ',');
			if(oneQuanta.length() == 0){
				continue; //I keep running into a miscounting problem with the last part
			}
			q.push_back(stoi(oneQuanta));
		}
		ket k(coeff, &q);
		groundStateKets->push_back(k);
		q.clear();
	}

	inputFile.close();
	cout << "Finished loading the old ground state." << endl;
}

void linearRegression(vector<double>* x, vector<double>* y){

        if (x->size() != y->size()){
                cout << "Warning: lists of x and y data are not the same size." << endl;
                return;
        }

        int n = x->size();

        double a; //The constant term
        double b; //The slope

        double xy = 0; // Sum over x_i * y_i
        double xSum = 0;
        double ySum = 0;
        double xSquaredSum = 0;

        for (int i = 0; i < n; i++){
                xy += x->at(i) * y->at(i);
                xSum += x->at(i);
                ySum += y->at(i);
                xSquaredSum += (x->at(i))*(x->at(i));
        }

        b = (xSum*ySum / n - xy) / (xSum*xSum / n - xSquaredSum);
        a = (ySum - b*xSum) / n;
        cout << "The linear fit is " << a << " + " << b << " x." << endl;

        double sumResidualsSquared = 0;
        double sumMeanDiffSquared = 0;

        for (int i = 0; i < n; i++){
                sumMeanDiffSquared += (y->at(i) - ySum/n)*(y->at(i) - ySum/n);
                sumResidualsSquared += (y->at(i) - (a + b*x->at(i)))*(y->at(i) - (a + b*x->at(i)));
        }
        cout << "Rsquared = " << 1 - sumResidualsSquared / sumMeanDiffSquared;
}

void plotCorrelators(vector<ket>* groundState, string corrOutput){
	ofstream outputFile;
	outputFile.open(corrOutput);
	outputFile << "m0 = " << m0 << endl;
	outputFile << "mbar = " << mbar << endl;
	outputFile << "reach = " << reach << endl;

	vector<ket> Rstate;
	vector<ket> Lstate;
	double result;

	vector<double> xs;
	vector<double> ys;

	for (int i = 1; i <= reach; i++){

		Rstate = *groundState;
		Lstate = *groundState;

		Rstate = correlator(&Rstate, i);
		result = innerProduct(&Lstate, &Rstate);
		if (result != 0){
			xs.push_back(i);
			ys.push_back(log(result));
		}
		outputFile << i << endl;
		outputFile << result << endl;
		cout << i << endl;
		cout << result << endl;
	}

	outputFile.close();

	linearRegression(&xs, &ys);
}

//check if a sparse matrix is diagonal
bool isSymmetric(Eigen::SparseMatrix<double>* Htrunc){

	int n = Htrunc->rows();
	if (n != Htrunc->cols()){
		cout << "Warning: matrix is not diagonal." << endl;	
	}

	for (int i = 0; i < n; i++){
		for (int j = 0; j <= i; j++){ //I only need to check the upper-triangular part of the matrix, so just go until j < i
			if ( abs(Htrunc->coeffRef(i,j) - Htrunc->coeffRef(j,i)) > cutoff){ //See if this element isn't equal to its partner across the diagonal
				cout << "Asymmetry at (" << i << "," << j << ")" << endl;
				cout << "Htrunc(" << i << "," << j << ") = " << Htrunc->coeffRef(i,j) << endl;
				cout << "Htrunc(" << j << "," << i << ") = " << Htrunc->coeffRef(j,i) << endl;
				cout << "Difference = " << Htrunc->coeffRef(i,j) - Htrunc->coeffRef(j,i) << endl;
				cout << "Relevant basis states are... " << endl;
				cout << "i = " << i << endl;
				cout << "j = " << j << endl;
				return false;
			}
		}
	}
	
	return true;	//If it made it through all that and never found one that didn't match, it's symmetric.
}

void testLadderOps(){
	vector<int> q1 = {0};
	vector<int> q2 = {1};
	vector<int> q3 = {3,1};
	vector<int> q4 = {2,0,4};
	ket k1(1,&q1);
	ket k2(1,&q2);
	ket k3(1,&q3);
	ket k4(1,&q4);
	vector<ket> state;
        state.push_back(k1);
        state.push_back(k2);
        state.push_back(k3);
        state.push_back(k4);

	cout << "State before acting on it: " << endl;
	listStateValues(&state);

	cout << "State after acting on it with a at 3:" << endl;
	aOp(&state,3);
	listStateValues(&state);
}

void testMergeKets(){
	vector<int> q1={1,0,1};
	ket k1(1,&q1);
	ket k2(-0.5,&q1);
	vector<int> q2 = {0};
	ket k3(0,&q2);
	vector<int> q4 = {2,3};
	ket k4(0.67,&q4);
	ket k5(-0.67,&q4);
	ket k6(2,&q1);
	
	vector<ket> state;
        state.push_back(k1);
        state.push_back(k2);
        state.push_back(k3);
        state.push_back(k4);
	state.push_back(k5);
	state.push_back(k6);
	
	cout << "Ket list before merging:" << endl;
	listStateValues(&state);
	cout << "Ket list after merging:" << endl;
	mergeKets(&state);
	listStateValues(&state);
}

void testPrune(){
	vector<int> q1 = {1,0,1};
	ket k1(1,&q1);
	vector<int> q2 = {2};
	ket k2(0, &q2);
	vector<int> q3 = {0};
	ket k3(pow(10,-14),&q3);
	vector<int> q4 = {1,1};
	ket k4(-0.5, &q4);

	vector<ket> state;
	state.push_back(k1);
	state.push_back(k2);
	state.push_back(k3);
	state.push_back(k4);

	cout << "States before pruning." << endl;
	listStateValues(&state);

	cout << "States after pruning." << endl;
	prune(&state);
	listStateValues(&state);
}

void testAdjustKetIndices(){

	vector<int> q = {1,1};
	ket k(1,&q);
	cout << "Ket before adjusting." << endl;
	k.listValues();

	int index = 1;
	cout << "Ket after adjusting to act on " << index << endl;
	int actIndex = adjustKetIndices(&k, index);
	cout << "The operator should act on index " << actIndex << endl;
	k.listValues();

}

void testKetRanking(){
	vector<int> q = {1,0,1};
        ket k1(0.67, &q);
        vector<int> q2 = {1,1};
        ket k2(-1, &q2);
        cout << "Ket ranking: " << rankTwoKets(&k1, &k2) << endl;
        cout << "Flipped ket ranking: " << rankTwoKets(&k2, &k1) << endl;
        cout << "If they're the same ket..." << rankTwoKets(&k1, &k1) << endl;

        vector<int> q3 = {2};
        ket k3(1, &q3);
        vector<int> q4 = {0};
        ket k4(1, &q4);
        vector<ket> unsortedState;
        unsortedState.push_back(k1);
        unsortedState.push_back(k2);
        unsortedState.push_back(k3);
        unsortedState.push_back(k4);
        vector<ket> sortedState = sortKetList(unsortedState);
        listStateValues(&sortedState);

}

void testBasisMaking(){
	vector<vector<ket>> useBasis;
        vector<ket> opBasis;
        vector<ket> matrixBasis;

        budget = 3;
        reach = 3;
        makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach);
        cout << "Complete basis given reach of " << reach << " and budget of " << budget << endl;
        listStateValues(&opBasis);
}

void testMakeHtrunc(){

	budget = 2;
	reach = 2;
	vector<vector<ket>> useBasis;
	vector<ket> opBasis;
	vector<ket> matrixBasis;

	makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach);
	cout << "Basis elements: " << endl;
	listStateValues(&opBasis);
	cout << endl;
	int basisSize = opBasis.size();

	Eigen::SparseMatrix<double> Htrunc(basisSize, basisSize);
	makeHtrunc(&Htrunc, &useBasis, &opBasis);

	cout << Htrunc << endl;

	vector<vector<ket>> groundStates;
	vector<double> groundEnergies;
	getGroundStates(&Htrunc, 1, &matrixBasis, &groundStates, &groundEnergies);
	cout << "Ground energy: " << groundEnergies.at(0) << endl;
	cout << "Ground state: " << endl;
	listStateValues(&(groundStates.at(0)));

	vector<ket> stateL = groundStates.at(0);
	vector<ket> stateR = groundStates.at(0);
	stateR = correlator(&stateR,1);

	cout << "stateR: " << endl;
	listStateValues(&stateR);
	cout << endl;

	cout << "stateL: " << endl;
	listStateValues(&stateL);
	cout << endl;

	cout << innerProduct(&stateL, &stateR) << endl;
}

//This is to see if Spectra is working and if I'm including the libraries properly
void testSpectra(){
	Eigen::MatrixXd A = Eigen::MatrixXd::Random(5,5);
	Eigen::MatrixXd M = A + A.transpose();

	cout << "Symmetric random matrix: " << endl;
	cout << M << endl;

	DenseSymMatProd<double> op(M);

	SymEigsSolver<DenseSymMatProd<double>> eigs(op, 1, 4);

	eigs.init();
	int nconv = eigs.compute(SortRule::SmallestAlge);

	Eigen::VectorXd evalues;
	if (eigs.info() == CompInfo::Successful)
		evalues = eigs.eigenvalues();

	cout << "Eigenvalues found:\n" << evalues << endl;
}

void checkSameQuanta(){
	vector<int> q1 = {0,2};
	vector<int> q2 = {1,1};
	vector<int> indices1 = {0,1};

	ket k1(1.0/sqrt(2), &q1);
	k1.indices = indices1;
	ket k2(-0.392699, &q2);
	k2.indices = indices1;

	cout << "ket 1: " << endl;
	k1.listValues();
	cout << "ket 2: " << endl;
	k2.listValues();
	cout << "Same quanta?" << endl;
	cout << sameQuanta(&k1, &k2) << endl;
}

void testRegression(){

	vector<double> x = {2,3,4,6};
	vector<double> y = {2,4,6,7};

	linearRegression(&x, &y);

}

void HactOnOneState(){
	
	vector<int> q1 = {0};
	vector<int> q2 = {1};
	vector<int> q3 = {1,1};
	vector<int> q4 = {1,1,1};
	vector<int> q5 = {1,1,0,1};
	vector<int> q6 = {1,0,1,1};

	ket k1(-0.986109,&q1);
	ket k2(-0.915997,&q2);
	ket k3(-0.14707,&q3);
	ket k4(-0.24345,&q4);
	ket k5(-0.140871,&q5);
	ket k6(-0.140871,&q6);

	vector<ket> state0;
	vector<ket> state1;

	state0.push_back(k3);
	cout << "Hop acting on |1,1>" << endl;

	vector<ket> result = HSumOverLSites(state0,-5,5);
	listStateValues(&result);
}

int main(int argc, char *argv[]){
	/*
	string filename = "r12FivePercent";
	makeHtruncConverge(0.05, 1, 12, filename);
	cout << "Finished making the truncated H." << endl;
	vector<ket> groundState;
	loadGroundStateFromFile(&groundState, filename + "GroundStates.txt");
	plotCorrelators(&groundState, filename + "Correlator.txt");
	*/
	//testRegression();

	vector<vector<ket>> useBasis;
	vector<ket> opBasis;
	vector<ket> matrixBasis;
	makeBasis(&useBasis, &opBasis, &matrixBasis, 2, 2);
	cout << "Made a basis of size " << opBasis.size() << endl;
	listStateValues(&opBasis);

	vector<ket> testState = useBasis.at(1);
	vector<ket> result = Hop(testState,-1);
	listStateValues(&result);

	/*
	int basisSize = opBasis.size();
	Eigen::SparseMatrix<double> Htrunc(basisSize, basisSize);

	makeHtrunc(&Htrunc, &useBasis, &opBasis);

	cout << "Is it symmetric? " << isSymmetric(&Htrunc) << endl;

        vector<vector<ket>> groundStates;
        vector<double> groundEnergies;
        getGroundStates(&Htrunc, 2, &matrixBasis, &groundStates, &groundEnergies);
        cout << "Ground energy: " << groundEnergies.at(1) << endl;
        cout << "Ground state: " << endl;
        listStateValues(&(groundStates.at(1)));
	cout << "Next-lowest state:" << endl;
	cout << "Energy = " << groundEnergies.at(0) << endl;
	cout << "State: " << endl;
	listStateValues(&(groundStates.at(0)));
	*/
}
