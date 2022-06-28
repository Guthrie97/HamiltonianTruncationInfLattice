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
#include "stdafx.h"
#include "interpolation.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace Spectra;
using namespace alglib;

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
			cout <<  "Basis Tag: " << basisTag << endl;
		}

		int totalQuanta(){
			int runningTotal = 0;
			for (int i = 0; i < quanta.size(); i++){
				runningTotal += quanta.at(i);
			}
			return runningTotal;
		}

		void resetIndices(){
			vector<int> updatedIndices;
			for (int i = 0; i < quanta.size(); i++){
				updatedIndices.push_back(i);
			}
			indices = updatedIndices;
		}

		void writeValues(ofstream outputFile){
			outputFile << coeff << endl;
			for (int i = 0; i < quanta.size(); i++){
				outputFile << quanta.at(i) << ",";
			}
			outputFile << endl;
			for (int i = 0; i < indices.size(); i++){
				outputFile << indices.at(i) << ",";
			}
			outputFile << endl;
			outputFile << basisTag << endl;
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

//Given a list of kets, sort them into even and odd. It preserves whatever original ordering they had.
void sortKetsByParity(vector<ket>* allStates, vector<ket>* evenStates, vector<ket>* oddStates){

	for (int i = 0; i < allStates->size(); i++){
		if ( (allStates->at(i)).totalQuanta() % 2 == 0 ){
			evenStates->push_back(allStates->at(i));
		} else{
			oddStates->push_back(allStates->at(i));
		}
	}

	//redo the basis numberings.
	for (int i = 0; i < evenStates->size(); i++){
		(evenStates->at(i)).basisTag = i;
	}
	for (int i = 0; i < oddStates->size(); i++){
		(oddStates->at(i)).basisTag = i;
	}
}

//Do the same thing, but for the use basis
void sortUseBasisByParity(vector<vector<ket>>* allStates, vector<vector<ket>>* evenStates, vector<vector<ket>>* oddStates){

	for (int i = 0; i < allStates->size(); i++){
		if ( (allStates->at(i)).size() > 1 ){
			cout << "Warning: more than one ket per state here. Quitting the parity sort now." << endl;
			return;
		}
		if ( (allStates->at(i)).at(0).totalQuanta() % 2 == 0){
			evenStates->push_back(allStates->at(i));
		} else{
			oddStates->push_back(allStates->at(i));
		}
	}
	
	//Redo the basis numberings.
	for (int i = 0; i < evenStates->size(); i++){
		(evenStates->at(i)).at(0).basisTag = i;
	}
	for (int i = 0; i < oddStates->size(); i++){
		(oddStates->at(i)).at(0).basisTag = i;
	}
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
//If basisType = 0, then it returns the opBasis index of it.
//If basisType = 1, then it returns the matrixBasis index of it.
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
			return (opBasis->at(m)).basisTag; //Return its position in the MATRIX ordering, not the ordering in the op basis
		}
	}

	//If it made it through the search without finding a match, return -1 to say the ket isn't in the basis
	return -1;

}

//Same thing, but with more options
//If basisType = 0, then it returns the opBasis index of it.
//If basisType = 1, then it returns the matrixBasis index of it.
int findKetIndex(ket* k, vector<ket>* opBasis, int basisType){

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
			if (basisType == 0){
				return m; //Return the position in the opBasis.
			} else if (basisType == 1) {
                        	return (opBasis->at(m)).basisTag; //Return its position in the MATRIX ordering, not the ordering in the op basis
                	}
		}
        }

        //If it made it through the search without finding a match, return -1 to say the ket isn't in the basis
        return -1;
}

//Does a binary search, but about where to insert a new ket into opBasis
void insertKet(ket* k, vector<ket>* opBasis){

	int L = 0;
	int R = opBasis->size() - 1;
	int m;
	int compResult;
	int secondCompResult;
	bool keepGoing = true;

	//This case ends up working funnily if the ket belongs at the beginning or end, so check this up front
	if (rankTwoKets(k, &(opBasis->at(R))) == 0){
		opBasis->push_back(*k);
		return;
	}
	if (rankTwoKets(k, &(opBasis->at(L))) == 1){
		opBasis->insert(opBasis->begin(), *k);
		return;
	}

	while (keepGoing){
		m = (L+R)/2;
		compResult = rankTwoKets(k, &(opBasis->at(m)));
		if (compResult == 1){
			R = m-1; //This means k is less than the ket at m
		} else if (compResult == 0){//This means k is greater than the ket at m
			secondCompResult = rankTwoKets(k, &(opBasis->at(m+1)));
			if (secondCompResult == 0){//This means k is also greater than m+1 so it's not in the middle
				L = m+2;
			} else if(secondCompResult == 1){//Now we know k is right between m and m+1, so it should be pushed in at spot m+1
				keepGoing = false;
			}
		}
	}
	opBasis->insert(opBasis->begin() + m+1, *k);
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
	//return localIndices.at(localIndices.size() - 1);
	return localIndices.size() - 1; //If it's made it all the way to the end, it needs to act at on the last index spot, when all the 0s have been padded out.

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

//Insert newKets, in-order, to an already-sorted list. THIS CAN BE IMPROVED. First, it sees if the new ket is already in the list. If so, it just adds the coefficient. If not, it does the binary search again, but the version about how to insert a new ket.
void insertToSortedList(vector<ket>* sortedList, vector<ket>* newKets){
	int ketIndex;
	for (int i = 0; i < newKets->size(); i++){
		ketIndex = findKetIndex( &(newKets->at(i)), sortedList, 0);
		if (ketIndex != -1){
			(sortedList->at(ketIndex)).coeff += (newKets->at(i)).coeff;
			if ( (sortedList->at(ketIndex)).coeff < cutoff){
				sortedList->erase(sortedList->begin() + ketIndex);
			}
		} else{
			insertKet( &(newKets->at(i)), sortedList);
		}
	}
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
void phiSquaredNO(vector<ket>* term1, int x){

	vector<ket> term2 = *term1;
	vector<ket> term3 = *term1;

	aOp(term1, x);
	aOp(term1, x);

	aDagOp(&term2, x);
	aDagOp(&term2, x);

	aOp(&term3, x);
	aDagOp(&term3, x);
	multiplyOverKets(&term3, 2);

	mergeKetLists(&term2, &term3);
	mergeKetLists(term1, &term2);

	multiplyOverKets(term1, 0.5/mbar);
}

//The cross-term operator phi_x phi_x+a
void phixphixpa(vector<ket>* term1, int x, int a){

	vector<ket> term2 = *term1;
	vector<ket> term3 = *term1;
	vector<ket> term4 = *term1;

	aOp(term1, x);
	aOp(term1, x+a);

	aDagOp(&term2, x);
	aDagOp(&term2, x+a);

	//For terms like these, always lead with the lowering operator. Starting with raising might push it out of budget, and it gets cut prematurely.
	aOp(&term3, x);
	aDagOp(&term3, x+a);

	aOp(&term4, x+a);
	aDagOp(&term4, x);

	mergeKetLists(&term3, &term4);
	mergeKetLists(&term2, &term3);
	mergeKetLists(term1, &term2);

	multiplyOverKets(term1, 0.5 / mbar);
}

//This is the translationally-invariant sum of the phi_x phi_x+a operator. I make it act on one ket at a time since the trial ground state has to be handled differently... but I could probably make this run faster.
void correlator(vector<ket>* state, int a){
	cout << "a = " << a << endl;

	vector<ket> result;
	vector<ket> tempResult;
	int ketSize;

        //First, find the trial ground state, pull it out, and handle it separately since it's already translationally invariant.
        for (int i = 0; i < state->size(); i++){
                if ( isTrialGroundState(&(state->at(i))) ){
                        ket k = state->at(i);
                        state->erase(state->begin() + i); //Pull the ket out and erase it from the rest
                        vector<ket> tempState;
                        tempState.push_back(k);
                        phixphixpa(&tempState,0,a);
                        mergeKetLists(&result, &tempState);
                        break;
                }
        }

	//Have the operator act on the other kets in the list one at a time. This is faster because I can re-size the translationally-invariant sum for each one, based on the limits with the reach
	for (int i = 0; i < state->size(); i++){
		if (i % 100 == 0){
			cout << "i = " << i << endl;
		}
		vector<ket> preppedState;
		preppedState.push_back(state->at(i));
		ketSize = (state->at(i)).quanta.size();
		for (int x = -(reach - ketSize); x < (reach - a); x++){
			tempResult = preppedState;
/*			if (i % 500 == 0){
				cout << "x = " << x << endl;
				cout << "Number of kets before operator: " << tempResult.size() << endl;
				listStateValues(&tempResult);
			}
*/			phixphixpa(&tempResult,x,a);
/*			if (i % 500 == 0){
				cout << "Number of kets after operator: " << tempResult.size() << endl;
				listStateValues(&tempResult);
			}
*/			insertToSortedList(&result, &tempResult);
/*			if (i % 500 == 0){
				cout << "Finished merging these results into the list." << endl;
			}
*/		}
	}

	/*
	//First, find the trial ground state, pull it out, and handle it separately since it's already translationally invariant.
	for (int i = 0; i < state->size(); i++){
		if ( isTrialGroundState(&(state->at(i))) ){
			tempKet k = state->at(i);
			state->erase(state->begin() + i); //Pull the ket out and erase it from the rest
			vector<ket> tempState;
			tempState.push_back(k);
			phixphixpa(&tempState,0,a);
			mergeKetLists(&result, &tempState);
			break;
		}
	}

	//Do a translationally-invariant sum over the rest
	vector<ket> preppedState;
	for (int x = -(reach - ketSize); x < (reach - 1); x++){
		preppedState = *state;
		phixphixpa(&preppedState,x,a);
		mergeKetLists(&result, &preppedState);
		preppedState.clear();
	}*/
	*state = result;
}

//Just the base QHO Hamiltonian, with the constant term dropped, acting on site x
void H0op(vector<ket>* state, int x){

	aOp(state, x);
	aDagOp(state, x);
	multiplyOverKets(state, mbar);
}

//The potential terms for the Hamiltonian, 1/2 (m^2 - mbar^2)phi_x^2 + 1/2 (phi_x+1 - phi_x)^2
void Vop(vector<ket>* term1, int x){

	vector<ket> term2 = *term1;
	vector<ket> term3 = *term1;

	phiSquaredNO(term1, x);
	multiplyOverKets(term1, 0.5*(m0*m0 - mbar*mbar + 1));

	phiSquaredNO(&term2, x+1);
	multiplyOverKets(&term2, 0.5);

	phixphixpa(&term3, x, 1);
	multiplyOverKets(&term3, -1);

	mergeKetLists(&term2, &term3);
	mergeKetLists(term1, &term2);
}

//The complete Hamiltonian density operator
void Hop(vector<ket>* term1, int x){

	vector<ket> term2 = *term1;

	H0op(term1, x);
	Vop(&term2, x);

	mergeKetLists(term1, &term2);
}

void HSumOverLSites(vector<ket>* state, int start, int finish){

	vector<ket> oneTermResult;
	vector<ket> allResults;

	for (int i = start; i < finish; i++){
		oneTermResult = *state;
		Hop(&oneTermResult, i);
		mergeKetLists(&allResults, &oneTermResult);
	}

	*state = allResults;
	//return allResults;

}

//check if a sparse matrix is diagonal
bool isSymmetric(Eigen::SparseMatrix<double>* Htrunc){

        int n = Htrunc->rows();
        if (n != Htrunc->cols()){
                cout << "Warning: matrix is not square." << endl;
        }

	Eigen::SparseMatrix<double> Htranspose = Htrunc->transpose();
	Eigen::SparseMatrix<double> matrixSum = *Htrunc - Htranspose;
	//Iterate over the elements explicitly listed and see if any are nonzero
	for (int k = 0; k < matrixSum.outerSize(); ++k){
		for (Eigen::SparseMatrix<double>::InnerIterator it(matrixSum, k); it; ++it){
			if (it.value() != 0){
				cout << "Htrunc is asymmetric at (" << it.row() << "," << it.col() << ")" << endl;
				cout << "Htrunc(" << it.row() << "," << it.col() << ") = " << Htrunc->coeffRef(it.row(), it.col()) << endl;
				cout << "Htrunc(" << it.col() << "," << it.row() << ") = " << Htrunc->coeffRef(it.col(), it.row()) << endl;
				return false;
			}
		}
	}

	/*
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
        }*/

        return true;    //If it made it through all that and never found one that didn't match, it's symmetric.
}


//Creates the sparse matrix for the whole Hamiltonian density
void makeHtrunc(Eigen::SparseMatrix<double>* Htrunc, vector<vector<ket>>* useBasis, vector<ket>* opBasis, int L){

	//L has to do with the size of the lattice and is important for the normalization of the non-trial-vacuum states.
	double matrixElementCoeff1 = 1.0 / L;
	double matrixElementCoeff2 = 1.0 / sqrt(L);

	cout << "Making Htrunc." << endl;

	vector<ket> preppedState;
	int row;
	int column;
	int startTransInvCounter;
	//Sum over all possible ways the translationally invariant states can be acted on by the Hamiltonian density.
	//But I accomplish this by holding the states still and moving the Hamiltonian density
	
	//Handle the trial ground state differently since it's already translationally invariant.
	if (isTrialGroundState(&(useBasis->at(0).at(0)))){
		startTransInvCounter = 1; //Start the next cycle from 1, since I already took care of 0
		preppedState = useBasis->at(0);
		column = preppedState.at(0).basisTag;
		Hop(&preppedState, 0);
		for (int j = 0; j < preppedState.size(); j++){
			row = findKetIndex(&(preppedState.at(j)), opBasis);
			//cout << "(row, column) = (" << row << "," << column << ")" << endl;
			if (row != -1){
				Htrunc->coeffRef(row,column) += preppedState.at(j).coeff * matrixElementCoeff2;
				//cout << "Updated the matrix." << endl;
			}
		}
	} else{
		startTransInvCounter = 0;
	}

	cout << "Done with the part with the trial ground ket." << endl;

	for (int i = startTransInvCounter; i < useBasis->size(); i++){
		preppedState = useBasis->at(i);
		if (preppedState.size() > 1){
			cout << "Warning: there are multiple kets in this basis state " << i << endl;
		}
		column = preppedState.at(0).basisTag;
		HSumOverLSites(&preppedState, -reach, reach);
		for (int j = 0; j < preppedState.size(); j++){
			row = findKetIndex(&(preppedState.at(j)), opBasis);
			//cout << "(row, column) = (" << row << "," << column << ")" << endl;
			if (row != -1){ //If row == -1, the ket is out of the basis
				Htrunc->coeffRef(row, column) += preppedState.at(j).coeff * matrixElementCoeff1;
			}
		}
	}
}

//Gets the n lowest many eigenvalues and eigenvectors from a sparse matrix. There's an option for picking the eigenvectors that use only states with an even number of excitations, which we believe is physical for the ground state
//It is recommended to ask for the two lowest eigenstates so we can pick the even one.
void getGroundStates(Eigen::SparseMatrix<double>* Htrunc, int numValues, vector<ket>* matrixBasis, vector<vector<ket>>* groundStates, vector<double>* groundEnergies, bool pickEven){
	//Follow the tutorial from Spectra
	SparseSymMatProd<double> op(*Htrunc);
	int basisSize = matrixBasis->size();
	int convergenceParameter = min(10*numValues, basisSize / 2 + 1);
	SymEigsSolver<SparseSymMatProd<double>> eigs(op, numValues, convergenceParameter);
	eigs.init();
	Eigen::VectorXd evalues;
	Eigen::MatrixXd evectors;
	int nconv = eigs.compute(SortRule::SmallestAlge);
	if (eigs.info() == CompInfo::Successful){
		evalues = eigs.eigenvalues();
		evectors = eigs.eigenvectors();
	} else if (eigs.info() == CompInfo::NotConverging){
		cout << "Warning: the eigenstuff did NOT converge." << endl;
	} else if (eigs.info() == CompInfo::NumericalIssue){
		cout << "Warning: There's some kind of numerical issue." << endl;
	} else {
		cout << "Warning: the eigenstuff was NOT computed successfully, and I don't understand why." << endl;
	}

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
		if (pickEven){//check the first ket to see if it's an even or odd number of excitations
			if ((eigenstate.at(0)).totalQuanta() % 2 == 1){
				continue; //If the first ket is odd, that means they all are. Don't save this and move on.
			}
		}
		eigenstate = sortKetsByCoeff(eigenstate);
		groundStates->push_back(eigenstate);
		groundEnergies->push_back(evalues(i)); //Note that this scheme means the lowest eigenstate is at the back of the vector.
	}

	if (groundEnergies->size() == 0){
		cout << "Warning: no eigenstates meeting the specifications were found." << endl;
	}
}

//The last parameter designates whether to expand the basis by fixing a reach and expanding the budget (0) or fixing the budget and expanding the reach (1).
//It seems like a long reach is more important than a high budget.
void makeHtruncConverge(double threshold, int numValues, int maxLength, int maxBudget, int L,  string outputFile, int parameterOption){

	//Open the output file
	ofstream energiesOutput;
	energiesOutput.open(outputFile + ".txt");

	//Set up the initial parameters
	if (parameterOption == 0){
		budget = 2;
		reach = maxLength;
	} else if(parameterOption == 1){
		reach = 6;
		budget = maxBudget;
	} else if (parameterOption == 2){
		budget = 2;
		reach = 6;
	}

	vector<vector<ket>> useBasis;
	vector<ket> opBasis;
	vector<ket> matrixBasis;
	makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach);

	//Sort out just the states with even numbers of excitations. Keep the odds around in case we'll use them later.
	vector<vector<ket>> evenUseBasis;
	vector<vector<ket>> oddUseBasis;
	vector<ket> evenOpBasis;
	vector<ket> oddOpBasis;
	vector<ket> evenMatrixBasis;
	vector<ket> oddMatrixBasis;

	sortKetsByParity(&opBasis, &evenOpBasis, &oddOpBasis);
	sortKetsByParity(&matrixBasis, &evenMatrixBasis, &oddMatrixBasis);
	sortUseBasisByParity(&useBasis, &evenUseBasis, &oddUseBasis);

	int basisSize = evenOpBasis.size();
	cout << "Size of the basis: " << basisSize << endl;

	Eigen::SparseMatrix<double> HtruncEven(basisSize, basisSize);
        makeHtrunc(&HtruncEven, &evenUseBasis, &evenOpBasis, 3*reach+3);

        vector<vector<ket>> groundStates;
        vector<double> groundEnergies;
        getGroundStates(&HtruncEven, numValues, &evenMatrixBasis, &groundStates, &groundEnergies, true);
	double prevGroundEnergy = groundEnergies.at(groundEnergies.size() - 1);
	cout << "Basis size: " << basisSize << endl;
	cout << "Ground Energy: " << prevGroundEnergy << endl;
	energiesOutput << basisSize << endl;
	energiesOutput << prevGroundEnergy << endl;
	double nextGroundEnergy;

	//Now that I have an initial value, I can start incrementing the budget and compare for convergence
	bool converged;
	while (!converged){
		//Increment the relevant parameter.
		if (parameterOption == 0){
			budget += 2; // Increment the budget by two, because of how the ground state only uses 1 parity.
		} else if (parameterOption == 1){
			reach += 2;
		} else if (parameterOption == 2){
			budget += 2;
			reach += 2; //Increment both by two if the convergence seems to be happening more slowly
		}
		useBasis.clear();
		opBasis.clear();
		matrixBasis.clear();
		evenUseBasis.clear();
		oddUseBasis.clear();
		evenOpBasis.clear();
		oddOpBasis.clear();
		evenMatrixBasis.clear();
		oddMatrixBasis.clear();

		makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach); //Make the new basis
	        sortKetsByParity(&opBasis, &evenOpBasis, &oddOpBasis);
	        sortKetsByParity(&matrixBasis, &evenMatrixBasis, &oddMatrixBasis);
	        sortUseBasisByParity(&useBasis, &evenUseBasis, &oddUseBasis);

		basisSize = evenOpBasis.size();
		cout << "Made a basis of size " << basisSize << " with reach " << reach << " and budget " << budget << endl;

		HtruncEven.setZero();
		HtruncEven.resize(basisSize, basisSize);
		makeHtrunc(&HtruncEven, &evenUseBasis, &evenOpBasis, 3*reach+3); //Make Htrunc again
		cout << "Made the truncated Hamiltonian." << endl;

		groundStates.clear();
		groundEnergies.clear();
		getGroundStates(&HtruncEven, numValues, &evenMatrixBasis, &groundStates, &groundEnergies, true);
		nextGroundEnergy = groundEnergies.at(groundEnergies.size() - 1);

		cout << "Basis size: " << basisSize << endl;
		cout << "Ground Energy: " << nextGroundEnergy << endl;
		energiesOutput << basisSize << ", " << budget << ", " << reach << ", " << 3*reach+3 << endl;
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
				for (int j = 0; j < state.size(); j++){
					statesOutput << (state.at(j)).coeff << endl;
					localQuanta = (state.at(j)).quanta;
					for (int k = 0; k < localQuanta.size(); k++){
						statesOutput << localQuanta.at(k) << ",";
					}
					statesOutput << endl;
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

//This function computes a number to characterize how different two states are. First, for all kets they share in common, it sums the difference of their coefficients squared. Then, for each ket unique to a state, its coefficient squared is added to the total too.
double computeStateDifferencesSquared(vector<ket> state1, vector<ket> state2, bool altPhase){

	//Sort the lists so it's easy to find matches between them.
	state1 = sortKetList(state1);
	state2 = sortKetList(state2);

	//Judge which one is shorter.
	vector<ket> shorterState = state1;
	vector<ket> longerState = state2;
	if (shorterState.size() > state2.size()){
		shorterState = state2;
		longerState = state1;
	}

	ket testKet = shorterState.at(0);
	int matchedKetIndex;
	//Make sure the states are in-phase.
	bool inPhase = false;
	int counter = 0;
	/*
	cout << "About to do the phase adjustment." << endl;
	while (!inPhase){
		testKet = shorterState.at(counter);
		matchedKetIndex = findKetIndex(&testKet, &longerState, 0);
		if (matchedKetIndex != -1){ //If we have a matched pair, then we can make sure they're on the same phase
			//See if multiplying one of the coefficients by -1 results in the difference between them being smaller
			if ( abs(testKet.coeff + longerState.at(matchedKetIndex).coeff ) > abs(testKet.coeff - longerState.at(matchedKetIndex).coeff ) ){
				multiplyOverKets(&longerState, -1); //In that case, multiply the longer one by -1.
				inPhase = true;
			}
		}
		counter++;
		if (counter == shorterState.size()){
			break;
		}
	}
*/
        double runningTotal = 0;
        double coeffDiff;
        while (shorterState.size() > 0) {
                testKet = shorterState.at(0);
                matchedKetIndex = findKetIndex(&testKet, &longerState, 0);
		if (matchedKetIndex != -1){
                	coeffDiff = testKet.coeff - longerState.at(matchedKetIndex).coeff;
                	runningTotal += coeffDiff * coeffDiff;
			longerState.erase(longerState.begin() + matchedKetIndex);
		} else{
			runningTotal += testKet.coeff * testKet.coeff;
		}
		shorterState.erase(shorterState.begin());
        }
	cout << "coeffDiff after going through just one state: " << runningTotal << endl;
	for (int i = 0; i < longerState.size(); i++){
		runningTotal += (longerState.at(i).coeff)*(longerState.at(i).coeff);
	}
	cout << "coeffDiff after finishing off the longer state: " << runningTotal << endl;

	cout << "Now trying this with the other phase:" << endl;
	multiplyOverKets(&state2, -1);
	if (!altPhase){
		double runningTotalAltPhase = computeStateDifferencesSquared(state1, state2, true);
		if (runningTotalAltPhase < runningTotal){
			runningTotal = runningTotalAltPhase;
		}
	}

	return runningTotal;
}


void makeGroundStateConverge(double threshold, int numValues, int maxLength, int maxBudget, int L,  string outputFile, int parameterOption){

        //Open the output file
        ofstream coeffDiffsOutput;
        coeffDiffsOutput.open(outputFile + ".txt");
	coeffDiffsOutput << "m0=" << m0 << endl;
	coeffDiffsOutput << "mbar=" << mbar <<endl;
	coeffDiffsOutput << "L=" << L << endl;

        //Set up the initial parameters
        if (parameterOption == 0){
                budget = 2;
                reach = maxLength;
        } else if(parameterOption == 1){
                reach = 2;
                budget = maxBudget;
        } else if (parameterOption == 2){
                budget = 2;
                reach = 6;
        }

        vector<vector<ket>> useBasis;
        vector<ket> opBasis;
        vector<ket> matrixBasis;
        makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach);

        //Sort out just the states with even numbers of excitations. Keep the odds around in case we'll use them later.
        vector<vector<ket>> evenUseBasis;
        vector<vector<ket>> oddUseBasis;
        vector<ket> evenOpBasis;
        vector<ket> oddOpBasis;
        vector<ket> evenMatrixBasis;
        vector<ket> oddMatrixBasis;

        sortKetsByParity(&opBasis, &evenOpBasis, &oddOpBasis);
        sortKetsByParity(&matrixBasis, &evenMatrixBasis, &oddMatrixBasis);
        sortUseBasisByParity(&useBasis, &evenUseBasis, &oddUseBasis);

        int basisSize = evenOpBasis.size();
        cout << "Size of the basis: " << basisSize << endl;

        Eigen::SparseMatrix<double> HtruncEven(basisSize, basisSize);
        makeHtrunc(&HtruncEven, &evenUseBasis, &evenOpBasis, L);

        vector<vector<ket>> groundStates;
        vector<double> groundEnergies;
        getGroundStates(&HtruncEven, numValues, &evenMatrixBasis, &groundStates, &groundEnergies, true);
	vector<ket> prevGroundState = groundStates.at(groundStates.size() - 1);
	vector<ket> nextGroundState;
	double coeffDiffTotal;

        //Now that I have an initial value, I can start incrementing the parameters and compare for convergence
        bool converged;
        while (!converged){
                //Increment the relevant parameter.
                if (parameterOption == 0){
                        budget += 2; // Increment the budget by two, because of how the ground state only uses 1 parity.
                } else if (parameterOption == 1){
                        reach += 2;
                } else if (parameterOption == 2){
                        budget += 2;
                        reach += 2; //Increment both by two if the convergence seems to be happening more slowly
                }
                useBasis.clear();
                opBasis.clear();
                matrixBasis.clear();
                evenUseBasis.clear();
                oddUseBasis.clear();
                evenOpBasis.clear();
                oddOpBasis.clear();
                evenMatrixBasis.clear();
                oddMatrixBasis.clear();

                makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach); //Make the new basis
                sortKetsByParity(&opBasis, &evenOpBasis, &oddOpBasis);
                sortKetsByParity(&matrixBasis, &evenMatrixBasis, &oddMatrixBasis);
                sortUseBasisByParity(&useBasis, &evenUseBasis, &oddUseBasis);

                basisSize = evenOpBasis.size();
                cout << "Made a basis of size " << basisSize << " with reach " << reach << " and budget " << budget << endl;

                HtruncEven.setZero();
                HtruncEven.resize(basisSize, basisSize);
                makeHtrunc(&HtruncEven, &evenUseBasis, &evenOpBasis, L); //Make Htrunc again
                cout << "Made the truncated Hamiltonian." << endl;

                groundStates.clear();
                groundEnergies.clear();
                getGroundStates(&HtruncEven, numValues, &evenMatrixBasis, &groundStates, &groundEnergies, true);
                nextGroundState = groundStates.at(groundStates.size() - 1);
		coeffDiffTotal = computeStateDifferencesSquared(prevGroundState, nextGroundState, false);

                cout << "Basis size: " << basisSize << endl;
                cout << "CoeffDiff: " << coeffDiffTotal << endl;
                coeffDiffsOutput << basisSize << ", " << budget << "," << reach << endl;
                coeffDiffsOutput << coeffDiffTotal << endl;

                if (coeffDiffTotal < threshold){ //If they've converged, finish the process

			converged = true;
                        //Now write the kets in the ground state to a file for later use
                        string outputFileGS = outputFile + "GroundStates.txt";
                        ofstream statesOutput;
                        statesOutput.open(outputFileGS);
                        //Immediately put in the key variables so those can be loaded later.
                        statesOutput << "m0=" << m0 << endl;
                        statesOutput << "mbar=" << mbar << endl;
                        statesOutput << "reach=" << reach << endl;
			statesOutput << "budget=" << budget << endl;
			statesOutput << "L=" << L << endl;
                        vector<ket> state;
                        vector<int> localQuanta;
                        double normTotal;
                        for (int i = 0; i < groundStates.size(); i++){
                                state = groundStates.at(groundStates.size() - 1 - i);
                                statesOutput << i << "," << groundEnergies.at(groundStates.size() - 1 - i) << endl;
                                for (int j = 0; j < state.size(); j++){
                                        statesOutput << (state.at(j)).coeff << endl;
                                        localQuanta = (state.at(j)).quanta;
                                        for (int k = 0; k < localQuanta.size(); k++){
                                                statesOutput << localQuanta.at(k) << ",";
                                        }
                                        statesOutput << endl;
                                }
                                statesOutput << "x" << endl; //This is a signal for the ket reader functions to move on
                        }
                        statesOutput.close();
                } else{
                        prevGroundState = nextGroundState;
                }
        }

        coeffDiffsOutput.close();
}

void setReachFromNewBasis(vector<ket>* ketList){

	int maxLength = 0;
	for (int i = 0; i < ketList->size(); i++){
		if ( (ketList->at(i)).quanta.size() > maxLength){
			maxLength = (ketList->at(i)).quanta.size();
		}
	}

	reach = maxLength;
}

double loadGroundEnergyFromFile(string filename){
	ifstream inputFile;
	inputFile.open(filename);

	string oneLine;
	inputFile >> oneLine;
	inputFile >> oneLine;
	inputFile >> oneLine; //Get through these lines, which are data we don't need
	inputFile >> oneLine;
	inputFile >> oneLine;
	return stod(oneLine.substr(2));
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
	inputFile >> oneLine;
	int L;
	L = stod(oneLine.substr(2));

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

void groundStatesDifferenceSquared(string input1, string input2){

	//Load all ground states
	vector<ket> groundState1;
	vector<ket> groundState2;
	loadGroundStateFromFile(&groundState1, input1);
	loadGroundStateFromFile(&groundState2, input2);

	//Sort them so I can compare states in them faster.
	groundState1 = sortKetList(groundState1);
	groundState2 = sortKetList(groundState2);

	ket testKet = groundState1.at(0);
	int matchedKetIndex;
	double runningTotal = 0;
	double coeffDiff;
	for (int i = 0; i < groundState1.size(); i++){
		testKet = groundState1.at(i);
		matchedKetIndex = findKetIndex(&testKet, &groundState2, 0);
		coeffDiff = testKet.coeff - groundState2.at(matchedKetIndex).coeff;
		runningTotal += coeffDiff * coeffDiff;
	}
	cout << runningTotal << endl;
}

void binKetCoeffs(string inputFile){

	vector<ket> groundState;
	loadGroundStateFromFile(&groundState, inputFile);
	cout << "Number of kets in the ground state: " << groundState.size() << endl;
	vector<int> coeffCounts = {0,0,0,0,0,0,0,0,0,0};
	double logCoeff;
	int bin;
	for (int i = 0; i < groundState.size(); i++){
		logCoeff = -log10(abs(groundState.at(i).coeff));
		bin = floor(logCoeff);
		while (bin >= coeffCounts.size()){
			coeffCounts.push_back(0);
		}
		coeffCounts.at(bin) += 1;
	}
	for (int i = 0; i < 10; i++){
		cout << "Bin " << i << " = " << coeffCounts[i] << endl;
	}
}

void outputKetCoeffs(string inputFile, string outputFileName){

	vector<ket> groundState;
	loadGroundStateFromFile(&groundState, inputFile);
	ofstream outputFile;
	outputFile.open(outputFileName);

	for (int i = 0; i < groundState.size(); i++){
		outputFile << groundState.at(i).coeff << endl;
	}

	outputFile.close();
}

//Given a ground state, it picks out only the kets with the largest coefficients until a certain fraction of the normalization is covered
void cutKetsByTotalNorm(double normalizationTarget, vector<ket>* masterGroundState, vector<ket>* cutGroundState, vector<vector<ket>>* useBasis, vector<ket>* opBasis, vector<ket>* matrixBasis){

	double runningTotal = 0;
	int ketCounter = 0;
	ket tempKet = masterGroundState->at(0);
	while (runningTotal < normalizationTarget){
		tempKet = masterGroundState->at(ketCounter);
		runningTotal += (tempKet.coeff)*(tempKet.coeff); //Add its normalization contribution to the running total
		cutGroundState->push_back(tempKet);
		//Now clean up the ket parameters and plug it into the basis
		tempKet.coeff = 1;
		tempKet.basisTag = ketCounter;
		tempKet.resetIndices();
		opBasis->push_back(tempKet);
		ketCounter++;
	}

	//Sort the opBasis then plug those into the other bases.
	*opBasis = sortKetList(*opBasis);
	*matrixBasis = *opBasis;
	for (int i = 0; i < opBasis->size(); i++){
		vector<ket> tempState;
		tempState.push_back(opBasis->at(i));
		useBasis->push_back(tempState);
	}
}

void cutKetsByTotalNormPlotEnergies(string inputFile, string outputName, int L){

	vector<ket> masterGroundState;
	vector<ket> cutGroundState;
	loadGroundStateFromFile(&masterGroundState, inputFile);
	double masterGroundEnergy = loadGroundEnergyFromFile(inputFile);

	ofstream outputFile;
	outputFile.open(outputName);

	ofstream altOutputFile;
	altOutputFile.open("basisDumps.txt");

	double normalizationTarget = 0;
	double runningTotal = 0;
	int ketCounter = 0;
	int numTotalKets = masterGroundState.size();
	vector<vector<ket>> useBasis;
	vector<ket> opBasis;
	vector<ket> matrixBasis;
	for (int i = 0; i < 5; i++){ //Counts how many 9s I put on my normalization target
		normalizationTarget += 9*pow(10,-1-i);

		cutKetsByTotalNorm(normalizationTarget, &masterGroundState, &cutGroundState, &useBasis, &opBasis, &matrixBasis);

		cout << "ketCounter = " << ketCounter << endl;
		cout << "opBasis.size() = " << opBasis.size() << endl;
		Eigen::SparseMatrix<double> Htrunc(opBasis.size(), opBasis.size());
		makeHtrunc(&Htrunc, &useBasis, &opBasis, L);
		vector<vector<ket>> groundStates;
		vector<double> groundEnergies;
		getGroundStates(&Htrunc, 1, &matrixBasis, &groundStates, &groundEnergies, true);
		outputFile << normalizationTarget << endl;
		outputFile << ((double) ketCounter) / ((double) numTotalKets) << endl;
		outputFile << groundEnergies.at(0) << endl;
		cout << normalizationTarget << endl;
		cout << ((double) ketCounter) / ((double) numTotalKets) << endl;
		cout << groundEnergies.at(0) << endl;
		cout << "groundEnergies.size() = " << groundEnergies.size() << endl;
	}

	outputFile.close();
	altOutputFile.close();
}

void sumNormByBudgetAndReach(string inputName, string outputName){
	vector<ket> masterGroundState;
	loadGroundStateFromFile(&masterGroundState, inputName);
	ofstream outputFile;
	outputFile.open(outputName);

	vector<double> budgetNorms(7, 0.0);
	vector<double> reachNorms(10, 0.0);
	ket tempKet = masterGroundState.at(0);
	int ketBudget;
	int ketSize;
	for (int i = 0; i < masterGroundState.size(); i++){
		tempKet = masterGroundState.at(i);
		ketBudget = tempKet.totalQuanta();
		ketSize = tempKet.quanta.size();
		budgetNorms.at(ketBudget/2) += tempKet.coeff * tempKet.coeff;
		reachNorms.at(ketSize-1) += tempKet.coeff * tempKet.coeff;
	}

	outputFile << "Budget norms: " << endl;
	cout << "Budget norms: " << endl;
	for (int i = 0; i < budgetNorms.size(); i++){
		outputFile << i*2 << ": " << budgetNorms.at(i) << endl;
		cout << i*2 << ": " << budgetNorms.at(i) << endl;
	}

	outputFile << "Reach norms: " << endl;
	cout << "Reach norms: " << endl;
	for (int i = 0; i < reachNorms.size(); i++){
		outputFile << i+1 << ": " << reachNorms.at(i) << endl;
		cout << i+1 << ": " << reachNorms.at(i) << endl;
	}
}

void loadBasisFromFile(vector<vector<ket>>* useBasis, vector<ket>* opBasis, vector<ket>* matrixBasis, string filename){

	ifstream inputFile;
	inputFile.open(filename);

	string line;
	vector<int> tempQuanta = {0};
	int i = 0;
	while (inputFile){
		inputFile >> line;//This reads in the coeff, which should be 1
		if (stoi(line) != 1){
			cout << "Warning: there's a basis element at " << i << " that wasn't entered with the right coefficient." << endl;
		}
		inputFile >> line; //This reads in the quanta
		stringstream quantaStream(line);
                string oneQuanta;
		vector<int> q;
                while (quantaStream.good()){
                        getline(quantaStream, oneQuanta, ',');
                        if(oneQuanta.length() == 0){
                                continue; //I keep running into a miscounting problem with the last part
                        }
                        q.push_back(stoi(oneQuanta));
                }
                ket k(1, &q);
		inputFile >> line; //This reads in the indices, which I don't need
		inputFile >> line; //This reads in the basis tag
		k.basisTag = stoi(line);

		//Now I'm ready to plug this into the bases
		opBasis->push_back(k);
		matrixBasis->push_back(k);


		i++;
	}
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
        cout << "Rsquared = " << 1 - sumResidualsSquared / sumMeanDiffSquared << endl;
}
/*
void function_cx_1_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr){
	func = exp(-c[0]*pow(x[0],2));
}

void TwoPtFitFunction(const real_1d_array &c, const real_1d_array &L, double &func, void *ptr){
	func = c[0] / pow(c[1]*L[0],0.5) * exp(-c[1]*L[0]);
}

void fitCorrelatorData(){
	real_2d_array x = "[[1],[2],[3],[4],[5],[6],[7]]";
	real_1d_array y = "[2.77988 1.99594 1.43685 0.991814 0.639843 0.371418 0.174103 0.0304461]";
	real_1d_array c = "[1.0, 1.0]";
	double epsx = 0.000001;
	ae_int_t maxits = 0;
	ae_int_t info;
	lsfitstate state;
	lsfitreport rep;
	double diffstep = 0.0001;

	lsfitcreatef(x,y,c,diffstep,state);
	lsfitsetcond(state, epsx, maxits);
	alglib::lsfitfit(state, TwoPtFitFunction);
	lsfitresults(state, info, c, rep);
	printf("%d\n", int(info));
	printf("%s\n", c.tostring(1).c_str());
}
*/
void plotCorrelators(vector<ket>* groundState, string corrOutput){
	ofstream outputFile;
	outputFile.open(corrOutput);
	outputFile << "m0=" << m0 << endl;
	outputFile << "mbar=" << mbar << endl;
	outputFile << "reach=" << reach << endl;

	vector<ket> Rstate;
	vector<ket> Lstate;
	double result;

	vector<double> xs;
	vector<double> ys;

	for (int i = 1; i <= reach; i++){

		Rstate = *groundState;
		Lstate = *groundState;

		correlator(&Rstate, i);
		cout << "i = " << i << ", computed the operator." << endl;
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

void linearRegressionFromFile(string fileName){

	ifstream inputFile;
	inputFile.open(fileName);
	//Read in the first three lines and do nothing; that's just parameter information
	string oneLine;
	inputFile >> oneLine;
	cout << oneLine << endl;
	inputFile >> oneLine;
	cout << oneLine << endl;
	inputFile >> oneLine;
	cout << oneLine << endl;

	//Keep reading in lines until done
	vector<double> xs;
	vector<double> ys;
	double xtemp;
	double ytemp;
	while (!inputFile.eof()){
		inputFile >> oneLine;
		xtemp = stod(oneLine);
		inputFile >> oneLine;
		ytemp = stod(oneLine);
		if (ytemp != 0){//If one of the ys is a 0, this'll mess up the log.
			xs.push_back(xtemp);
			ys.push_back(log(ytemp)); //Note that this assumes we're plotting a 2-pt function, so I need to take the log before doing a linear regression
		}
	}

	inputFile.close();

	linearRegression(&xs, &ys);
}

//Updates the operator and matrix basis, with the proper ordering with new basis elements. Ones to take out are in oldKets, and new ones are in newKets. Then, the matrix indices of each new ket is stored in matrixUpdateList
void updateBasis(vector<ket>* opBasis, vector<ket>* matrixBasis, vector<ket>* oldKets, vector<ket>* newKets, vector<int>* matrixUpdateList){

	if (oldKets->size() != newKets->size()){
		cout << "Warning: number of old kets to remove and new kets to replace them with don't match." << endl;
		return;
	}

	//First, go through each old ket. Give away its spot in the matrix basis to a new ket, and make sure the new ket has the basis tag that goes along with it. Then, delete the old ket from the opBasis and insert the new ket in the opBasis.
	int oldOpIndex;
	int matrixIndex;
	for (int i = 0; i < oldKets->size(); i++){
		//Matrix basis part
		matrixIndex = findKetIndex(&(oldKets->at(i)), opBasis);
		(newKets->at(i)).basisTag = matrixIndex;
		matrixBasis->at(matrixIndex) = newKets->at(i);
		matrixUpdateList->push_back(matrixIndex);

		//Op basis part
		oldOpIndex = findKetIndex(&(oldKets->at(i)), opBasis, 0);
		opBasis->erase(opBasis->begin() + oldOpIndex);
		insertKet(&(newKets->at(i)), opBasis);
	}

	//Update the reach, since that'll be important for searching for new kets later
	for (int i = 0; i < newKets->size(); i++){
		if ((newKets->at(i)).quanta.size() > reach){
			reach = (newKets->at(i)).quanta.size();
		}
	}
}

//Recalculates select rows and columns of the matrix based on matrixUpdateList
void updateMatrix(Eigen::SparseMatrix<double>* Htrunc, vector<ket>* opBasis, vector<ket>* matrixBasis, vector<int>* matrixUpdateList){
	vector<int> tempQuanta = {0};
	ket tempKet(1,&tempQuanta); //This just defines the variables so they aren't created new every loop
	vector<ket> preppedState;
	int row;
	int column;

	//I need to do a step here to clear out the rows and columns that'll be changed - they have to be zeroed out before I continue
	for (int i = 0; i < matrixUpdateList->size(); i++){
		for (int j = 0; j < opBasis->size(); j++){
			Htrunc->coeffRef(matrixUpdateList->at(i),j) = 0;
			Htrunc->coeffRef(j,matrixUpdateList->at(i)) = 0;
		}
	}

	cout << "Matrix after zeroing out appropriate rows and columns: " << endl;
	cout << *Htrunc << endl;

	for (int i = 0; i < matrixUpdateList->size(); i++){
		column = matrixUpdateList->at(i);
		tempKet = matrixBasis->at(column); //Pull up the ket that's been updated
		preppedState.push_back(tempKet);
		HSumOverLSites(&preppedState, -reach, reach); //Act on it with the translationally-invariant operator
		for (int j = 0; j < preppedState.size(); j++){
			tempKet = preppedState.at(j); //Go through each ket in the result, find their matrix index, and update
			row = findKetIndex(&tempKet, opBasis);
			if (row != -1){
				Htrunc->coeffRef(row, column) += tempKet.coeff;
				if (row != column){
					Htrunc->coeffRef(column, row) += tempKet.coeff;
				}
			}
		}
		preppedState.clear(); //Clear out the variable for the next pass through the loop
	}

	//cout << "Matrix after all updates: " << endl;
	//cout << *Htrunc << endl;

}

//Do a version of the gradient-descent method on the approximate ground state. Fills in lists of kets that will be used to update the basis.
void findNewKetsFromGroundState(vector<ket>* groundState, vector<ket>* opBasis, vector<ket>* matrixBasis, vector<ket>* oldKets, vector<ket>* newKets, double prevGroundEnergy, double groundStateCutoff){

	vector<ket> prevGroundState = *groundState;
	//Act on the previous ground state with	HSum... etc etc but make it one longer on either end
	HSumOverLSites(groundState, -(reach+1), reach+1);

	//Check every ket in the list against the previous ground state. If it can't be found, save it to the list of new kets.
	vector<ket> potentialNewKets;
	prevGroundState = sortKetList(prevGroundState); //Sort it so I can look up kets in this list, like the opBasis
	for (int i = 0; i < groundState->size(); i++){
		if (findKetIndex(&(groundState->at(i)), &prevGroundState) == -1){ //This means the ket isn't in the list
			potentialNewKets.push_back(groundState->at(i));
		}
	}
	potentialNewKets = sortKetsByCoeff(potentialNewKets); //Sort them by greatest coefficient to least to figure out which is most imporant.

	cout << "Potential new kets: " << endl;
	listStateValues(&potentialNewKets);

	//Check every ket in the opBasis against the previous ground state. If it can't be found, we can get rid of it, so save it to the list of old kets.
	//They're all equally unimportant, so don't worry about sorting them.
	vector<ket> potentialOldKets;
	for (int i = 0; i < opBasis->size(); i++){
		if (findKetIndex( &(opBasis->at(i)), &prevGroundState) == -1){
			potentialOldKets.push_back(opBasis->at(i));
		}
	}

	if (potentialOldKets.size() == 0){
		cout << "All the kets in the basis matter for the ground state. Time to judge which ones are the most interesting." << endl;
		//Collect the kets below a given cutoff. Hopefully the kets from the new list will matter more
		int numPrevKets = prevGroundState.size();
		ket tempKet = prevGroundState.at(0); //Just a placeholder
		int numNewKets = 0; //Counts how many new kets we're pulling in from potentialNewKets
		for (int i = 0; i < numPrevKets; i++){
			//Count backwards to save time; start from where the coeffs are low
			tempKet = prevGroundState.at(numPrevKets - 1 - i);
			if ( abs(tempKet.coeff) < groundStateCutoff){
				oldKets->push_back(tempKet); //This ket matters too little to the ground state; get rid of it
				newKets->push_back(potentialNewKets.at(numNewKets));
				numNewKets++;
			} else{
				break; //Anything else on the list is going to be too high too.
			}
		}
		if (numNewKets == 0){
			cout << "No new kets were added. Lower the cutoff." << endl;
		}
		return;
	}

	if (potentialNewKets.size() < potentialOldKets.size()){
		//All the potential new kets will go into the basis
		*newKets = potentialNewKets;
		for (int i = 0; i < potentialNewKets.size(); i++){
			oldKets->push_back(potentialOldKets.at(i));
		}
	} else{ //All the old kets will get swapped out, and only the new kets with the highest coefficients will go in
		*oldKets = potentialOldKets;
		for (int i = 0; i < potentialOldKets.size(); i++){
			newKets->push_back(potentialNewKets.at(i));
		}
	}

	//Reset the coefficients of the new kets
	for (int i = 0; i < newKets->size(); i++){
		(newKets->at(i)).coeff = 1;
		(newKets->at(i)).resetIndices();
	}
}

void setUserParameters(){
	cout << "Enter a whole number for the budget: " << endl;
	int tempBudget;
	cin >> tempBudget;
	budget = tempBudget;
	cout << "Enter a whole number for the reach: " << endl;
	int tempReach;
	cin >> tempReach;
	reach = tempReach;
	double convergenceThreshold;
	cout << "Enter a decimal number for the convergence threshold: " << endl;
	cin >> convergenceThreshold; //This isn't used anywhere yet; I just want to leave it here and modify it later.
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

	//budget = 2;
	//reach = 2;
	budget = 4;
	reach = 4;
	vector<vector<ket>> useBasis;
	vector<ket> opBasis;
	vector<ket> matrixBasis;

	//makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach);
	vector<int> q1 = {0};
	vector<int> q2 = {1,0,1};
	vector<int> q3 = {1,1};
	vector<int> q4 = {1,1,1,1};
	vector<int> q5 = {1,2,1};
	vector<int> q6 = {2};
	vector<int> q7 = {2,2};
	ket k1(1,&q1);
	ket k2(1,&q2);
	ket k3(1,&q3);
	ket k4(1,&q4);
	ket k5(1,&q5);
	ket k6(1,&q6);
	ket k7(1,&q7);
	opBasis.push_back(k1);
	opBasis.push_back(k2);
	opBasis.push_back(k3);
	opBasis.push_back(k4);
	opBasis.push_back(k5);
	opBasis.push_back(k6);
	opBasis.push_back(k7);

        matrixBasis.push_back(k1);
        matrixBasis.push_back(k5);
        matrixBasis.push_back(k3);
        matrixBasis.push_back(k2);
        matrixBasis.push_back(k6);
        matrixBasis.push_back(k4);
        matrixBasis.push_back(k7);

	for (int i = 0; i < matrixBasis.size(); i++){
		matrixBasis.at(i).basisTag = i;
	}
	vector<int> matrixTags = {0,3,2,5,1,4,6};
	for (int i = 0; i < opBasis.size(); i++){
		opBasis.at(i).basisTag = matrixTags.at(i);
		vector<ket> tempState;
		tempState.push_back(opBasis.at(i));
		useBasis.push_back(tempState);
	}

	cout << "Basis elements: " << endl;
	listStateValues(&opBasis);
	cout << endl;
	int basisSize = opBasis.size();

	Eigen::SparseMatrix<double> Htrunc(basisSize, basisSize);
	int L = 3*reach + 3;
	makeHtrunc(&Htrunc, &useBasis, &opBasis, L);

	cout << Htrunc << endl;

	vector<vector<ket>> groundStates;
	vector<double> groundEnergies;
	getGroundStates(&Htrunc, 2, &matrixBasis, &groundStates, &groundEnergies, false);
	cout << "Ground energy: " << groundEnergies.at(1) << endl;
	cout << "Ground state: " << endl;
	listStateValues(&(groundStates.at(1)));

	/*
	vector<ket> stateL = groundStates.at(0);
	vector<ket> stateR = groundStates.at(0);
	correlator(&stateR,1);

	cout << "stateR: " << endl;
	listStateValues(&stateR);
	cout << endl;

	cout << "stateL: " << endl;
	listStateValues(&stateL);
	cout << endl;

	cout << innerProduct(&stateL, &stateR) << endl;
	*/
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

	vector<double> x = {1,2,3,4,5,6,7};
	vector<double> y = {1.02240776, 0.69111512, 0.36245322, -0.00821969, -0.44653245, -0.99042717, -1.7481082};

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

	HSumOverLSites(&state0,-5,5);
	listStateValues(&state0);
}

void testInsertToSortedList(){
        vector<int> q1 = {0};
        vector<int> q2 = {1,1};
        vector<int> q3 = {1,0,1};
        vector<int> q4 = {2};
        reach = 3;
        ket k1(0.5, &q1);
        ket k2(0.5, &q2);
        ket k3(0.5, &q3);
        ket k4(0.5, &q4);
        vector<ket> cutGroundState;
        cutGroundState.push_back(k2);
        cutGroundState.push_back(k1);
        cutGroundState.push_back(k3);
        cutGroundState.push_back(k4);

        cutGroundState = sortKetList(cutGroundState);

        vector<ket> newKets;
        vector<int> q5 = {2,1,1};
        ket k5(0.2, &q5);
        newKets.push_back(k5);
        k1.coeff = 0.25;
        newKets.push_back(k1);
        k4.coeff = -0.5;
        newKets.push_back(k4);
        insertToSortedList(&cutGroundState, &newKets);
        listStateValues(&cutGroundState);

}

void testUpdateBasis(){

	double ketCutoff = 0.0001; //Just a placeholder

	//Make a simple basis
	budget = 3;
	reach = 2;
	vector<vector<ket>> useBasis;
	vector<ket> opBasis;
	vector<ket> matrixBasis;
	makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach);
	cout << "Starting basis:" << endl;
	listStateValues(&opBasis);
	
	//Make the Hamiltonian to go with it
	int basisSize = opBasis.size();
	Eigen::SparseMatrix<double> Htrunc(basisSize, basisSize);
	int L = 3*reach + 3;
	makeHtrunc(&Htrunc, &useBasis, &opBasis, L);
	cout << "Htrunc in the starting basis: " << endl;
	cout << Htrunc << endl;

	vector<vector<ket>> groundStates;
	vector<double> groundEnergies;
	getGroundStates(&Htrunc, 2, &matrixBasis, &groundStates, &groundEnergies, true);

	cout << endl;
	cout << "Energy = " << groundEnergies.at(0) << endl;
	listStateValues(&(groundStates.at(0)));
	cout << endl;
	//cout << "Energy = " << groundEnergies.at(1) << endl;
	//listStateValues(&(groundStates.at(1)));
	
	vector<ket> oldKets;
	vector<ket> newKets;
	findNewKetsFromGroundState(&(groundStates.at(0)), &opBasis, &matrixBasis, &oldKets, &newKets, groundEnergies.at(0), ketCutoff);
	cout << "Old kets to remove from the basis: " << endl;
	listStateValues(&oldKets);
	cout << endl;
	cout << "New kets to add to the basis" << endl;
	listStateValues(&newKets);

	vector<int> matrixUpdateList;
	updateBasis(&opBasis, &matrixBasis, &oldKets, &newKets, &matrixUpdateList);

	cout << "Updated basis (operator ordering): " << endl;
	listStateValues(&opBasis);

	cout << "Updated basis (matrix ordering): " << endl;
	listStateValues(&matrixBasis);

	cout << "Matrix indices of updated basis elements: " << endl;
	printVector(&matrixUpdateList);

	updateMatrix(&Htrunc, &opBasis, &matrixBasis, &matrixUpdateList);

	cout << "Updated matrix: " << endl;
	cout << Htrunc << endl;

	cout << "New ground state: " << endl;
	groundStates.clear();
	groundEnergies.clear();
	getGroundStates(&Htrunc, 1, &matrixBasis, &groundStates, &groundEnergies, false);
	listStateValues(&groundStates.at(0));

	cout << "Now putting that ket through the process one more time..." << endl;
	oldKets.clear();
	newKets.clear();
	findNewKetsFromGroundState(&(groundStates.at(0)), &opBasis, &matrixBasis, &oldKets, &newKets, groundEnergies.at(0), ketCutoff);

	//Update the basis by swapping out odds for evens
	/*vector<ket> oldKets;
	vector<ket> newKets;
	vector<int> matrixUpdateList;
	oldKets.push_back(opBasis.at(1));
	oldKets.push_back(opBasis.at(3));
	oldKets.push_back(opBasis.at(5));
        oldKets.push_back(opBasis.at(6));
	vector<int> newQuanta1 = {1,0,1};
	vector<int> newQuanta2 = {2,2};
	vector<int> newQuanta3 = {4};
	vector<int> newQuanta4 = {1,1,1,1};
	ket k1(1, &newQuanta1);
	ket k2(1, &newQuanta2);
	ket k3(1, &newQuanta3);
	ket k4(1, &newQuanta4);
	newKets.push_back(k1);
	newKets.push_back(k2);
	newKets.push_back(k3);
	newKets.push_back(k4);
	cout << "Old kets: " << endl;
	listStateValues(&oldKets);
	cout << "New kets: " << endl;
	listStateValues(&newKets);
	updateBasis(&opBasis, &matrixBasis, &oldKets, &newKets, &matrixUpdateList);
	cout << "New basis in the operator ordering: " << endl;
	listStateValues(&opBasis);
	cout << "New basis in the matrix ordering: " << endl;
	listStateValues(&matrixBasis);
	cout << "Matrix indices I need to update: " << endl;
	printVector(&matrixUpdateList);

	//Update the matrix
	updateMatrix(&Htrunc, &opBasis, &matrixBasis, &matrixUpdateList);
	cout << endl;
	cout << "Is it symmetric? " << isSymmetric(&Htrunc) << endl;
	*/
}

void testComputeStateDifferencesSquared(){

	vector<int> q1 = {0};
	vector<int> q2 = {1,1};
	vector<int> q3 = {2,2};
	vector<int> q4 = {2};

	ket k1(0.5, &q1);
	ket k2(0.5, &q2);
	ket k3(0.5, &q3);
	ket k4(0.5, &q4);

	vector<ket> state1;
	state1.push_back(k1);
	state1.push_back(k2);
	state1.push_back(k3);
	state1.push_back(k4);

	vector<ket> state2;
	state2.push_back(k1);
	state2.push_back(k2);
	state2.push_back(k3);
	//state2.push_back(k4);
	vector<int> q5 = {3};
	ket k5(0.1, &q5);
	state2.push_back(k5);

	cout << computeStateDifferencesSquared(state1, state2, false) << endl;
}

void varyLcheckEnergy(){

	budget = 6;
	reach = 8;

        vector<vector<ket>> useBasis;
        vector<ket> opBasis;
        vector<ket> matrixBasis;
        makeBasis(&useBasis, &opBasis, &matrixBasis, budget, reach);
	vector<int> Ls = {27, 33, 39, 45, 51};

	vector<vector<ket>> groundStates;
	vector<double> groundEnergies;

	for (int i = 0; i < Ls.size(); i++){
		Eigen::SparseMatrix<double> Htrunc(opBasis.size(), opBasis.size());
		makeHtrunc(&Htrunc, &useBasis, &opBasis, Ls.at(i));
		getGroundStates(&Htrunc, 1, &matrixBasis, &groundStates, &groundEnergies, true);
		cout << "L = " << Ls.at(i) << endl;
		cout << "Ground state energy = " << groundEnergies.at(0) << endl;
	}
}

int main(int argc, char *argv[]){

	/*
	vector<int> testQuanta = {1};
	vector<int> testIndices = {1};
	ket k(-0.383897,&testQuanta);
	k.indices = testIndices;
	vector<ket> state;
	state.push_back(k);
	listStateValues(&state);
	aDagOp(&state, 2);
	listStateValues(&state);
	*/
	//vector<ket> result = phixphixpa(&state, 0, 2);
	//listStateValues(&result);

	//testUpdateBasis();

	//testComputeStateDifferencesSquared();
	
	varyLcheckEnergy();

	//setUserParameters();
	//listStateValues(&opBasis);
	string filename = "";
	//binKetCoeffs(filename);
	//cout << loadGroundEnergyFromFile(filename) << endl;

	//outputKetCoeffs("r10FivePercentEvenGroundStates.txt","r10FivePercentEvenGroundStatesCoeffs.txt");
	//sumNormByBudgetAndReach("r10FivePercentEvenGroundStates.txt", "r10FivePercentEvenBudgetAndReachNorms.txt");

	//cutKetsByTotalNormPlotEnergies(filename, "r10FivePercentEvenCutGroundEnergies.txt");
	//reach = 15;
	//budget = 6;
	//int L = reach * 3 + 3;
	//makeHtruncConverge(0.03, 1, reach, 8, L, filename, 1);
	//makeGroundStateConverge(0.001, 1, reach, budget, L, filename, 1);
	//vector<ket> groundState;
	//loadGroundStateFromFile(&groundState, filename + "GroundStates.txt");
	//vector<ket> cutGroundState;
	//double normCutThreshold = 0.9999;
	//cutKetsByTotalNorm(normCutThreshold, &groundState, &cutGroundState, &useBasis, &opBasis, &matrixBasis);
	//cout << cutGroundState.size() << endl;
	//setReachFromNewBasis(&cutGroundState);
	//cout << "New reach = " << reach << endl;
	
	//plotCorrelators(&cutGroundState, filename + "CorrelatorNormCut9999.txt");
	
	/*
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
