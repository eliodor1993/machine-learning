/// Fichier  : ClassifieIndividus.cpp
/// Author : Eliodor Ednalson Guy Mirlin
/// Date : 12/04/2017
/// Programme : TP1 classifie les individus Algo KNN

#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include<cstdlib>
#include <cmath>

//Parameters definition for FP base
#define FP_LINE_TRN 320
#define FP_COL_TRN 2001
#define FP_LINE_TEST 160
#define FP_COL_TEST 2001

using namespace std;

// Les paramètres en entrées
std::string trn_in;
std::string tst_in;
int k_in;

//Global variables
int nb_col_trn = 0;
int nb_line_trn = 0;
int nb_col_tst = 0;
int nb_line_tst = 0;
int nb_classes = 0;

// Definir le nombre de ligne et colonne
int nb_ligne_fichier_iris_trn=100;
int nb_ligne_fichier_iris_tst=50;
int nb_col_fichier_iris=5;

int nb_ligne_fichier_letter_trn=13334;
int nb_ligne_fichier_letter_tst=6666;
int nb_col_fichier_letter=17;

int nb_ligne_fichier_optics_trn=3823;
int nb_ligne_fichier_optics_tst=1797;
int nb_col_fichier_optics=65;


//Read data from an file

float **read_data(string file, string type) {
    int nb_line = 0, nb_col = 0;
    float **data;
    ifstream file_reader(file.c_str(), ios::in);
    if (file_reader) // opening of the file
    {

         //Defining number of line and column of data set
        if (file.compare("data/fp/fp.trn") != 0
                && type.compare("trn") == 0) {
            if(file.compare("data/iris/iris.trn")==0){
                nb_line_trn=nb_ligne_fichier_iris_trn;
                nb_col_trn=nb_col_fichier_iris;
            }else if (file.compare("data/letter/let.trn")==0){
                nb_line_trn=nb_ligne_fichier_letter_trn;
                nb_col_trn=nb_col_fichier_letter;
            }else if(file.compare("data/optics/opt.trn")==0){
                nb_line_trn=nb_ligne_fichier_optics_trn;
                nb_col_trn=nb_col_fichier_optics;
            }

            nb_line = nb_line_trn;
            nb_col = nb_col_trn;
        }

        if (file.compare("data/fp/fp.tst") != 0
                && type.compare("tst") == 0) {
            if(file.compare("data/iris/iris.tst")==0){
                nb_line_tst=nb_ligne_fichier_iris_tst;
                nb_col_tst=nb_col_fichier_iris;
            }else if (file.compare("data/letter/let.tst")==0){
                nb_line_tst=nb_ligne_fichier_letter_tst;
                nb_col_tst=nb_col_fichier_letter;
            }else if(file.compare("data/optics/opt.tst")==0){
                nb_line_tst=nb_ligne_fichier_optics_tst;
                nb_col_tst=nb_col_fichier_optics;
            }


            nb_line = nb_line_tst;
            nb_col = nb_col_tst;
        }


        if (file.compare("data/fp/fp.trn") == 0) {
            nb_line_trn = FP_LINE_TRN;
            nb_col_trn = FP_COL_TRN;
            nb_line = nb_line_trn;
            nb_col = nb_col_trn;
        }

        if (file.compare("data/fp/fp.tst") == 0) {
            nb_line_tst = FP_LINE_TEST;
            nb_col_tst = FP_COL_TEST;
            nb_line = nb_line_tst;
            nb_col = nb_col_tst;

        }

        // Creation of a matrix to store data
        data = new float* [nb_line];
        for (int i = 0; i < nb_line; i++) {
            data[i] = new float[nb_col];
        }

        //Storing of data in the matrix
        for (int k = 0; k < nb_line; k++) {
            for (int l = 0; l < nb_col; l++) {
                file_reader >> data[k][l];
            }

        }


        file_reader.close(); // closing of the file
    } else {
        cerr << "Erreur à l'ouverture !" << endl;
    }

    return data;
}

// Euclidian distance from a vector to another one

float dist_vec_to_vec(float vector1[], float vector2[]) {

    float dist = 0;
    for (int i = 0; i < nb_col_trn - 1; i++) {
        dist += (vector1[i] - vector2[i])*(vector1[i] - vector2[i]);
    }
    dist = sqrt(dist);
    return dist;
}

//Manhattan distance from a vector to another one

float dist_vec_to_vec_man(float vector1[], float vector2[]) {

    float dist = 0;
    for (int i = 0; i < nb_col_trn - 1; i++) {
        dist += abs((vector1[i] - vector2[i]));
    }
    return dist;
}

//Distance from a vector to all the vectors of a matrix

float** dist_vec_to_set(float** set, float vector[]) {

    float** dist;
    dist = new float*[nb_line_trn];
    for (int i = 0; i < nb_line_trn; i++) {
        dist[i] = new float[2];
    }

    for (int i = 0; i < nb_line_trn; i++) {
        dist[i][0] = i;
        dist[i][1] = dist_vec_to_vec(set[i], vector);
    }
    return dist;
}

//Comparing two rows

bool compareTwoRows(float* rowA, float* rowB) {
    return (rowA[1] < rowB[1]);
}

//Class prediction by k-means algorithm

float k_means(float** set, float vector[], int k) {

    //distance from a test vector to the trainig et of vector
    float ** tab = dist_vec_to_set(set, vector);

    //sorting of the distance array obtained
    std::sort(tab, tab + nb_line_trn, &compareTwoRows);

    //storage of k first results classes
    float* knn;
    knn = new float[k];

    for (int i = 0; i < k; i++) {
        knn[i] = set[(int) tab[i][0]][nb_col_trn - 1];
    }
    std::sort(knn, knn + k);

    //Determination of the number of classes
    int * compteur;
    compteur = new int[k];
    for (int i = 0; i < k; i++) {
        compteur[i] = 0;
    }

    for (int i = 0; i < k; i++) {

        compteur[i] = std::count(knn, knn + k, knn[i]);
        i = i + compteur[i] - 1;
    }

    //Determination of the class with the biggest occurence
    int max_index = 0;
    for (int m = 1; m < k; m++) {
        if (compteur[max_index] < compteur[m]) {
            max_index = m;
        }
    }
    float classe = knn[max_index];

    return classe;
}

//computing of the classification performance

float bon_taux(int **conf_mat) {
    float taux = 0;
    float somme_diag = 0;
    float somme_tot = 0;

    //sum of diagonal elements is divided by sum of
    //all the matrix elements
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

void hold_out() {
    system("sort -R data/fp/fp.data > data/fp/fp.txt");
    system("head -n320 data/fp/fp.txt > data/fp/fp.trn");
    system("tail -n160 data/fp/fp.txt > data/fp/fp.tst");
}

int main() {

        cout<<endl<<" ----------- \033[1;31mClassificateur K N N ( k plus proches voisins )\033[0m\ ---------------" <<endl<<endl;
    do{

        // entrer le nom du fichier de la base d’apprentissage
        cout<< "entrer le nom du fichier de la base d’apprentissage : ";
        cin >> trn_in ; //
        // entrer le nom du fichier de la base de test
        cout<< "entrer le nom du fichier de la base de test : ";
        cin >> tst_in; //
        // k (le nombre de plus proches voisins)
        cout<< "entrer k (les plus proches voisins): ";
        cin >> k_in;

        cout<<endl;

        if((trn_in == "")&&(tst_in =="")&&(k_in==0)){
				cout << "Entrer les Fichier Requise"<<endl;
			}
        else {
        int k;
        k  = k_in;

        float** training;
        float** test;

        //Reading of data from specific base
        string hold_out_file(trn_in);
        if (hold_out_file.compare("data/fp/fp.trn") == 0) {
            hold_out();
       }

        training = read_data(trn_in, "trn");
        test = read_data(tst_in, "tst");

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
        int *prediction;
        prediction = new int[nb_line_tst];
        for (int i = 0; i < nb_line_tst; i++) {
            prediction[i] = (int) k_means(training, test[i], k);
        }

        //construction of the confusion matrix
        for (int m = 0; m < nb_line_tst; m++) {
            int pos1 = std::find(classes.begin(), classes.end(), (int) test[m][nb_col_tst - 1]) - classes.begin();
            int pos2 = std::find(classes.begin(), classes.end(), (int) prediction[m]) - classes.begin();
            confusion_matrix[pos1][pos2]++;
        }

        //Printing of number of classes, confusion matrix and performance
        cout << "nombre de classes : " << nb_classes << endl<<endl;
        cout << "La Matrice De Confusion  "<<endl;
        for (int a = 0; a < nb_classes; a++) {
            for (int b = 0; b < nb_classes; b++) {

                cout << confusion_matrix[a][b] << " ";
            }
            cout << endl<<endl;

        }

        cout << "le taux de précision est de : " << bon_taux(confusion_matrix)*100 << " %" << endl;


        return 0;

    }

    }while(trn_in!="");
}
