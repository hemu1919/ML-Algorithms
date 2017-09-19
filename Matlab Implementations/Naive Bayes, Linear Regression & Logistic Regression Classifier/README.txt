Steps to execute the software was:
 1.) First execute loadAllData.m, which will loads the datasets, their corresponding labels and other required variables
like parameters, number of folds, number of word_id's to work on, into the current workspace.
			>> loadAllData;
 2.) Next run the getSets.m, which takes the training data and number of folds as inputs and produces the number of sets 
equal to the number of folds, where each set consists of the record_id's (doc_id's) that comes into the particular set. It
randomly splits the given examples into the corresponding number of sets.
			>> r_sets=getSets(train_data,folds);
 3.) Now execute trainClassifier, with the following inputs, training data, training labels, number of sets of record id's
count of word_id's to work on, parameters and options. The options value takes either '1' (Naive Bayes Classifier) or 
'2' (Decision Tree Classifier) or '3' (Logistic Regression Classifier). If options value being '1' or '3' it will call
on crossVaidate to obtain the model using held out cross validation technique. Otherwise, it first computes the word_id's
equal to the count we passed by calling selectAttribute and then calls crossValidate.m
			>> [score,model]=trainClassifier(train_data,train_label,r_sets,w_count,params,options);
 4.) To test on the testing data set, call the corresponding functions:
	i.) for Naive Bayes Classification :
			>> label=bayesClassifier(test_data,model);
	ii.) for decision Tree Classification :
			>> label=decisionTreeClassifier(test_data,model);
	iii.) for Logisitic Regression Classification :
			>> label=logisticRegressionClassifier(test_data,model);
 5.) One more function is regularize_param.m, which takes training data, training labels, record_id's, count of word_ids, 
parameter set and options as inputs. The options values are same as said in setp (3). This function iteratively calls the 
trainClassifier using appropriate option value upon each value in the parameter set and collects the accuracy and average-
accuracy of the model using that parameter value and finally plots the graph.
			>> regularize_param(train_data,train_label,r_sets,w_count,params_set,options);