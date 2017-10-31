/// Fichier  : ClassifieIndividus.cpp
/// Author : ELIODOR Ednalson Guy Mirlin
/// Date : 12/04/2017
/// Programme : TP2.1 _perceptron_simple


#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>

//Parameters definition for ovarian and spam base

#define SPAM_LINE_TRN 3068
#define SPAM_COL_TRN 59
#define SPAM_LINE_TEST 1533
#define SPAM_COL_TEST 59

#define OVARIAN_LINE_TRN 169
#define OVARIAN_COL_TRN 15156
#define OVARIAN_LINE_TEST 84
#define OVARIAN_COL_TEST 15156

#define ALLAML_LINE_TRN 38
#define ALLAML_LINE_TST 34
#define ALLAML_COL 7130

#define BASE_LINE_TRN 23
#define BASE_LINE_TST 23
#define BASE_COL 4
using namespace std;


// Les paramètres en entrées
std::string trn_in;
std::string tst_in;
double step;
int max_iter;

string database;

//Global variables
int nb_col_trn = 0;
int nb_line_trn = 0;
int nb_col_tst = 0;
int nb_line_tst = 0;
int nb_classes = 0;

//Read data from an
//Read data from an file
double **read_data(string file, string type) {
    int nb_line = 0, nb_col = 0;
    double **data;
    ifstream file_reader(file.c_str(), ios::in);
    if (file_reader) // opening of the file
    {

        //Defining number of line and column of data set
        if (file.compare("data/spam/spam.trn") != 0 && file.compare("data/ovarian/ovarian.trn") != 0 && file.compare("data/base.trn") != 0
                && type.compare("trn") == 0) {
		if(file.compare("data/leukemia/ALLAML.trn")==0){
                nb_line_trn=ALLAML_LINE_TRN;
                nb_col_trn=ALLAML_COL;
            }
            //file_reader >> nb_line_trn >> nb_col_trn;
            nb_col_trn++;
            nb_line = nb_line_trn;
            nb_col = nb_col_trn;
        }

        if (file.compare("data/spam/spam.tst") != 0 && file.compare("data/ovarian/ovarian.tst") != 0 && file.compare("data/base.tst") != 0
                && type.compare("tst") == 0) {
		if(file.compare("data/leukemia/ALLAML.tst")==0){
		file.erase(file.begin()+13, file.end());
                nb_line_tst=ALLAML_LINE_TST;
                nb_col_tst=ALLAML_COL;
            }
            //file_reader >> nb_line_tst >> nb_col_tst;
            nb_col_tst++;
            nb_line = nb_line_tst;
            nb_col = nb_col_tst;
        }

        if (file.compare("data/spam/spam.trn") == 0) {
            nb_line_trn = SPAM_LINE_TRN;
            nb_col_trn = SPAM_COL_TRN;
            nb_line = nb_line_trn;
            nb_col = nb_col_trn;
        }

        if (file.compare("data/spam/spam.tst") == 0) {
            nb_line_tst = SPAM_LINE_TEST;
            nb_col_tst = SPAM_COL_TEST;
            nb_line = nb_line_tst;
            nb_col = nb_col_tst;
	    file.erase(file.begin()+9, file.end());

        }

        if (file.compare("data/ovarian/ovarian.trn") == 0) {
            nb_line_trn = OVARIAN_LINE_TRN;
            nb_col_trn = OVARIAN_COL_TRN;
            nb_line = nb_line_trn;
            nb_col = nb_col_trn;
        }

        if (file.compare("data/ovarian/ovarian.tst") == 0) {
            nb_line_tst = OVARIAN_LINE_TEST;
            nb_col_tst = OVARIAN_COL_TEST;
            nb_line = nb_line_tst;
            nb_col = nb_col_tst;
	    file.erase(file.begin()+12, file.end());

        }

 	 if (file.compare("data/base.trn") == 0) {
            nb_line_trn = BASE_LINE_TRN;
            nb_col_trn = BASE_COL;
            nb_line = nb_line_trn;
            nb_col = nb_col_trn;
        }

        if (file.compare("data/base.tst") == 0) {
            nb_line_tst = BASE_LINE_TST;
            nb_col_tst = BASE_COL;
            nb_line = nb_line_tst;
            nb_col = nb_col_tst;
	    file.erase(file.begin()+9, file.end());

        }


        // Creation of a matrix to store data
        data = new double* [nb_line];
        for (int i = 0; i < nb_line; i++) {
            data[i] = new double[nb_col];
        }

        //we assign 1 to all elements of the first column
        //this value will  be used to multiply the bias

        for (int i = 0; i < nb_line; i++) {
            data[i][0] = 1;
        }

         //Storing of data in the matrix
        for (int k = 0; k < nb_line; k++) {
            for (int l = 1; l < nb_col; l++) {
                file_reader >> data[k][l];
            }
        }

        file_reader.close(); // closing of the file
    } else {
        cerr << "Erreur à l'ouverture !" << endl;
    }

	database = (file);
    return data;
}

//Comparing two rows
bool compareTwoRows(double* rowA, double* rowB) {
    return (rowA[1] < rowB[1]);
}

//Normalisation of data set

double ** normalize_data(double** data, int nb_line, int nb_col) {

    double* column;
    column = new double[nb_line];

    for (int i = 1; i < nb_col - 1; i++) {
        for (int j = 0; j < nb_line; j++) {

            column[j] = data[j][i];

        }
        std::sort(column, column + nb_line);
        for (int j = 0; j < nb_line; j++) {

            data[j][i] = (data[j][i] - column[0]) / (column[nb_line - 1] - column[0]);

        }
    }

    return data;
}

//Prediction of class from weight and inputs

int predictor(double * poids, double *line) {

    int predicted_class = 1;
    double sortie = 0;

    for (int i = 0; i < nb_col_trn - 1; i++) {
        sortie += poids[i] * line[i];
    }

    if (sortie < 0) {
        predicted_class = -1;
    }

    return predicted_class;
}

//computing of weigth that fit to the training set

double* perceptron(double** train_set, double step, int max_iter) {

    double *poids;
    bool good = false;
    int count = 0;
    poids = new double[nb_col_trn - 1];
    int predicted_classe;

    //assigning random values to weight at their initialisation
    for (int i = 0; i < nb_col_trn - 1; i++) {

        poids[i] = rand() / double(RAND_MAX);

    }

    //Perceptron algorithm
    for (int j = 0; j < max_iter; j++) {
        for (int k = 0; k < nb_line_trn; k++) {

            predicted_classe = predictor(poids, train_set[k]);

            if (predicted_classe != train_set[k][nb_col_trn - 1]) {
                count++;

                for (int i = 0; i < nb_col_trn - 1; i++) {

                    poids[i] += step * (train_set[k][nb_col_trn - 1] - predicted_classe) * train_set[k][i];

                }

            }
        }
    }

    return poids;
}

//computing of the classification performance

double bon_taux(int **conf_mat) {
    double taux = 0;
    double somme_diag = 0;
    double somme_tot = 0;

    for (int i = 0; i < nb_classes; i++) {
        for (int j = 0; j < nb_classes; j++) {
            if (i == j) {
                somme_diag += conf_mat[i][j];
            }
            somme_tot += conf_mat[i][j];
        }
    }
    taux = somme_diag / somme_tot;
    return taux;

}

//Hold-out operation

void hold_out(string file) {

    //hold-out of spam base
    if (file.compare("data/spam/spam.trn") == 0) {
        system("sort -R data/spam/spam.data > data/spam/spam.txt");
        system("head -n3068 data/spam/spam.txt > data/spam/spam.trn");
        system("tail -n1533 data/spam/spam.txt > data/spam/spam.tst");

    } else { //hold-out of ovarian base

        system("sort -R data/ovarian/ovarian.data > data/ovarian/ovarian.txt");
        system("head -n169 data/ovarian/ovarian.txt > data/ovarian/ovarian.trn");
        system("tail -n84 data/ovarian/ovarian.txt > data/ovarian/ovarian.tst");
    }

}

int main() {

        cout<<endl<<" ----------- \033[1;31mImplémenter de l’algorithme perceptron\033[0m\ ---------------" <<endl<<endl;
    do{

        // entrer le nom du fichier de la base d’apprentissage
        cout<< "entrer le nom du fichier de la base d’apprentissage : ";
        cin >> trn_in ; //
        // entrer le nom du fichier de la base de test
        cout<< "entrer le nom du fichier de la base de test : ";
        cin >> tst_in; //

        cout<< "entrer le pas d’apprentissage: ";
        cin >> step;
        // k (le nombre de plus proches voisins)
        cout<< "entrer l'iterstion maximal): ";
        cin >> max_iter;

        cout<<endl;

        if((trn_in == "")&&(tst_in =="")){
				cout << "Entrer les Fichier Requise"<<endl;
			}
        else {


    double** training;
    double** test;


    //Reading of data from specific base
    string hold_out_file(trn_in);
    if (hold_out_file.compare("data/ovarian/ovarian.trn") == 0 ||
            hold_out_file.compare("data/spam/spam.trn") == 0) {
        hold_out(trn_in);
    }
    training = read_data(trn_in, "trn");
    test = read_data(tst_in, "tst");

    //Normalisation of data set for spam and leukemia base
    if (hold_out_file.compare("data/leukemia/ALLAML.trn") == 0 ||
            hold_out_file.compare("data/spam/spam.trn") == 0) {

        training = normalize_data(training, nb_line_trn, nb_col_trn);
        test = normalize_data(test, nb_line_tst, nb_col_tst);
    }

    //Storage of expected classes
    int supposed_classes [nb_line_tst];
    for (int i = 0; i < nb_line_tst; i++) {
        supposed_classes[i] = test[i][nb_col_tst - 1];
    }

    //Determination of number of expected classes
    std::sort(supposed_classes, supposed_classes + nb_line_tst);
    int compt = 0;
    std::vector<int> classes;
    for (int i = 0; i < nb_line_tst; i++) {

        compt = std::count(supposed_classes, supposed_classes +
                nb_line_tst, supposed_classes[i]);
        classes.push_back(supposed_classes[i]);
        i = i + compt - 1;
        nb_classes++;

    }

    //creation of the confusion matrix
    int **confusion_matrix;
    confusion_matrix = new int*[nb_classes];
    for (int j = 0; j < nb_classes; j++) {
        confusion_matrix[j] = new int [nb_classes];
        for (int n = 0; n < nb_classes; n++) {
            confusion_matrix[j][n] = 0;
        }
    }

    //computing of predited classes
    double* poids;
    poids = new double[nb_col_trn - 1];
    poids = perceptron(training, step, max_iter);

    int *prediction;
    prediction = new int[nb_line_tst];
    for (int i = 0; i < nb_line_tst; i++) {
        prediction[i] = predictor(poids, test[i]);
    }

    //construction of the confusion matrix
    for (int m = 0; m < nb_line_tst; m++) {

        int pos1 = std::find(classes.begin(), classes.end(), (int) test[m][nb_col_tst - 1]) - classes.begin();
        int pos2 = std::find(classes.begin(), classes.end(), (int) prediction[m]) - classes.begin();
        confusion_matrix[pos1][pos2]++;
    }

    //Printing of number of classes, confusion matrix and performance
    cout << "nombre de classes :" << nb_classes << endl;

    for (int a = 0; a < nb_classes; a++) {
        for (int b = 0; b < nb_classes; b++) {
            cout << confusion_matrix[a][b] << " ";
        }
        cout << endl;
    }

    cout << "le taux de précision est :" << bon_taux(confusion_matrix)*100 << "%" << endl;


    return 0;

}
 }while(trn_in!="");

}
