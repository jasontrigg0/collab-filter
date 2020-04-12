#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include <cmath>
#include <random>
#include <set>
#include <map>

typedef std::tuple<
  std::map<int,std::string>,
  std::map<int,std::string>,
  int,
  int,
  std::vector<std::vector<int>>,
  std::vector<std::vector<int>>> preprocess_tuple;
typedef std::vector<std::vector<std::string>> dataframe;

template<typename T>
std::tuple<std::vector<T>, std::vector<T>> gen_train_test(std::vector<T> data, float training_prob) {
  std::vector<T> train;
  std::vector<T> test;
  for (T row : data) {
    float r = ((float) rand() / (RAND_MAX));
    if (r < training_prob) {
      train.push_back(row);
    } else {
      test.push_back(row);
    }
  }
  return std::tuple<std::vector<T>,
                    std::vector<T>>(train, test);
}

preprocess_tuple preprocess(std::vector<std::vector<std::string>> &data) {
  std::set<std::string> users;
  std::set<std::string> questions;
  for (std::vector<std::string> row: data) {
    std::string user = row[0];
    std::string question = row[1];
    std::string val   = row[2];
    users.insert(user);
    questions.insert(question);
  }
  std::vector<std::string> user_vec(users.begin(), users.end());
  std::vector<std::string> question_vec(questions.begin(), questions.end());

  std::map<int,std::string> idx2user;
  std::map<int,std::string> idx2question;
  std::map<std::string,int> user2idx;
  std::map<std::string,int> question2idx;
  for (int i=0; i<user_vec.size(); i++) {
    std::string u = user_vec[i];
    idx2user[i] = u;
    user2idx[u] = i;
  }
  for (int i=0; i<question_vec.size(); i++) {
    std::string q = question_vec[i];
    idx2question[i] = q;
    question2idx[q] = i;
  }

  std::vector<std::vector<std::string>> filtered_data;
  for (std::vector<std::string> row: data) {
    std::string user = row[0];
    std::string question = row[1];
    std::string val   = row[2];
    int user_id = user2idx[user];
    if (user_id < 1000) {
      filtered_data.push_back(row);
    }
  }

  std::tuple<std::vector<std::vector<std::string>>,
             std::vector<std::vector<std::string>>> out = gen_train_test<std::vector<std::string>>(filtered_data, 0.8);
  std::vector<std::vector<std::string>> train = std::get<0>(out);
  std::vector<std::vector<std::string>> test = std::get<1>(out);
  std::vector<std::vector<int>> train_int;
  std::vector<std::vector<int>> test_int;

  for (std::vector<std::string> row: train) {
    std::vector<int> out;
    out.push_back(user2idx[row[0]]);
    out.push_back(question2idx[row[1]]);
    out.push_back(std::stoi(row[2]));
    train_int.push_back(out);
  }
  for (std::vector<std::string> row: test) {
    std::vector<int> out;
    out.push_back(user2idx[row[0]]);
    out.push_back(question2idx[row[1]]);
    out.push_back(std::stoi(row[2]));
    test_int.push_back(out);
  }

  return preprocess_tuple(idx2user, idx2question,
                          user_vec.size(), question_vec.size(),
                          train_int, test_int);
}

void printData(std::vector<std::vector<int>> &data) {
  for (std::vector<int> row: data) {
    for (int field: row) {
      printf("%i\n",field);
    }
  }
}

void print_vector(std::vector<std::string> &v, std::ostream & os) {
  std::string out = "";
  for (std::string s : v) {
    out += s + ",";
  }
  out = out.substr(0,out.size() - 1); //drop the trailing comma
  os << out.c_str() << std::endl;
}

void print_2d_vector(std::vector<std::vector<std::string>> &v, std::ostream & os = std::cout) {
  for (std::vector<std::string> v2: v) {
    print_vector(v2, os);
  }
}

std::vector<std::vector<std::string>> loadCsv(std::string filename) {
  std::ifstream infile(filename);
  std::string line;
  std::vector<std::vector<std::string>> data;
  while (std::getline(infile, line)) {
    std::stringstream lineStream(line);
    std::string field;
    std::vector<std::string> row;
    while (std::getline(lineStream,field,',')) {
      row.push_back(field);
    }
    if (row.size() == 3 && row[2] != "val") {
      data.push_back(row);
    }
  }
  // printData(data);
  return data;
}

float pred(int rowId, int colId, float total_mean, std::vector<float> &col_means, std::vector<std::vector<float>> &row_features, std::vector<std::vector<float>> &col_features) {
  float out = total_mean + col_means[colId];
  for (int i=0; i<row_features[0].size(); i++) {
    out += row_features[rowId][i] * col_features[colId][i];
  }
  return out;
}

void ratingPca(std::string filename) {
  int dim = 5;
  int matrix[dim][dim];
  for (int i=0; i<dim; i++) {
    for (int j=0; j<dim; j++) {
      matrix[i][j] = i * j;
    }
  }

  // printf("%d\n",matrix[2][2]);

  printf("Loading csv...\n");
  std::vector<std::vector<std::string>> raw_data = loadCsv(filename);

  printf("Preprocessing data...\n");
  preprocess_tuple data = preprocess(raw_data);

  std::map<int, std::string> idx2user = std::get<0>(data);
  std::map<int, std::string> idx2question = std::get<1>(data);
  int row_cnt = std::get<2>(data);
  int col_cnt = std::get<3>(data);
  std::vector<std::vector<int>> train = std::get<4>(data);
  std::vector<std::vector<int>> test = std::get<5>(data);

  int feature_cnt = 1;

  std::vector<float> col_sums(col_cnt,0);
  std::vector<float> col_cnts(col_cnt,0);

  int total_cnt = 0;
  float total_sum = 0;

  printf("Computing column means...\n");
  for(std::vector<int> row: train) {
    int rowId = row[0];
    int colId = row[1];
    int val   = row[2];
    col_sums[colId] += val;
    col_cnts[colId]++;
    total_sum += val;
    total_cnt++;
  }

  float total_mean = total_sum / total_cnt;

  //setup col_means
  std::vector<float> col_means(col_cnt,0);
  for(int i = 0; i<col_sums.size(); i++) {
    if (col_cnts[i] > 0) {
      col_means[i] = (col_sums[i] / col_cnts[i]) - total_mean;
    } else {
      col_means[i] = 0.0;
    }
    // printf("%i,%f\n",i,col_means[i]);
  }

  //initialize features to 0 + random noise
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> d(0,0.01);

  std::vector<std::vector<float>> row_features;
  std::vector<std::vector<float>> col_features;

  printf("Initializing features...\n");
  for (int r=0; r<row_cnt; r++) {
    std::vector<float> row;
    for (int k=0; k<feature_cnt; k++) {
      row.push_back(d(gen));
    }
    row_features.push_back(row);
  }
  for (int c=0; c<col_cnt; c++) {
    std::vector<float> col;
    for (int k=0; k<feature_cnt; k++) {
      col.push_back(d(gen));
    }
    col_features.push_back(col);
  }

  // row_cnt,std::vector<float>(feature_cnt,0));
  // col_cnt,std::vector<float>(feature_cnt,0));

  int epoch_cnt = 10000;
  float lrate = 0.001; //0.01
  float lam = 15; //.1; //7;

  printf("Training...\n");
  for (int i=0; i<epoch_cnt; i++) {
    if (i % 100 == 0) {
      //compute training set error
      float sum_sq_err = 0;
      for (std::vector<int> row: train) {
        int rowId = row[0];
        int colId = row[1];
        int val   = row[2];
        float err = val - pred(rowId, colId, total_mean, col_means, row_features, col_features);
        sum_sq_err += pow(err,2);
      }
      float obj_fn = sum_sq_err;
      for (int k=0; k<feature_cnt; k++) {
        for (int c=0; c<col_features.size(); c++) {
          obj_fn += lam * pow(pow(col_features[c][k],2),1);
        }
        for (int r=0; r<row_features.size(); r++) {
          obj_fn += lam * pow(pow(row_features[r][k],2),1);
        }
      }
      printf("-----\n");
      printf("sum_sq_err: %f\n", sum_sq_err);
      printf("obj_fn: %f\n", obj_fn);

      //compute test set error
      float test_sum_sq_err = 0;
      for (std::vector<int> row: test) {
        int rowId = row[0];
        int colId = row[1];
        int val = row[2];
        float err = val - pred(rowId, colId, total_mean, col_means, row_features, col_features);
        test_sum_sq_err += pow(err, 2);
      }
      printf("test sum_sq_err: %f\n", test_sum_sq_err);
    }
    for (int k=0; k<feature_cnt; k++) {
      for (std::vector<int> row: train) {
        int rowId = row[0];
        int colId = row[1];
        int val   = row[2];
        float err = val - pred(rowId, colId, total_mean, col_means, row_features, col_features);
        float row_val = row_features[rowId][k];
        float col_val = col_features[colId][k];
        // if (i > 50000 && rowId == 0) {
        //   printf("%i,%i,%i,%f\n", rowId, colId, val, err);
        //   printf("%f,%f\n", row_val, col_val);
        //   printf("%f,%f\n", lrate * (err * col_val - lam * row_val), lrate * (err * row_val - lam * col_val));
        // }
        row_features[rowId][k] += lrate * (err * col_val);
        col_features[colId][k] += lrate * (err * row_val);
      }
      for (int rowId=0; rowId < row_features.size(); rowId++) {
        float row_val = row_features[rowId][k];
        row_features[rowId][k] -= lrate * lam * row_val;
        // if (row_val != 0) {
        //   row_features[rowId][k] -= (row_val > 0 ? 1 : -1) * lam;
        // }
      }
      for (int colId=0; colId < col_features.size(); colId++) {
        float col_val = col_features[colId][k];
        col_features[colId][k] -= lrate * lam * col_val;
        // if (col_val != 0) {
        //   col_features[colId][k] -= (col_val > 0 ? 1 : -1) * lam;
        // }
      }
    }
  }

  // printf("col means:\n");
  // print_vector(col_means);
  // printf("row features:\n");
  // print_2d_vector(row_features);
  // printf("col features:\n");
  // print_2d_vector(col_features);


  printf("Writing to csv...\n");

  //write to csv
  std::vector<std::string> user_cols;
  user_cols.push_back("user");
  for (int k=0; k<feature_cnt; k++) {
    user_cols.push_back("feature_" + std::to_string(k));
  }
  dataframe df_users;
  df_users.push_back(user_cols);
  for (int i=0; i<row_features.size(); i++) {
    std::vector<std::string> row;
    row.push_back(idx2user[i]);
    for (int k=0; k<feature_cnt; k++) {
      row.push_back(std::to_string(row_features[i][k]));
    }
    df_users.push_back(row);
  }

  std::vector<std::string> question_cols;
  question_cols.push_back("question");
  for (int k=0; k<feature_cnt; k++) {
    question_cols.push_back("feature_" + std::to_string(k));
  }
  dataframe df_questions;
  df_questions.push_back(question_cols);
  for (int i=0; i<col_features.size(); i++) {
    std::vector<std::string> row;
    row.push_back(idx2question[i]);
    for (int k=0; k<feature_cnt; k++) {
      row.push_back(std::to_string(col_features[i][k]));
    }
    df_questions.push_back(row);
  }

  std::ofstream of1("/tmp/b.csv");
  print_2d_vector(df_users, of1);

  std::ofstream of2("/tmp/c.csv");
  print_2d_vector(df_questions, of2);
}

int main(int argc, const char** argv) {
  printf("helloworld\n");

  std::string csvFile = "/home/jtrigg/files/okcupid/sample_data_test.csv";

  if (argc > 1) {
    csvFile = std::string(argv[1]);
  }

  ratingPca(csvFile);
}
