All Files and descriptions in the HW3.zip are:

  1.) loaddata - A script file which loads the required .csv files and call the nntrain call with the training data with its labels and then evaluate the model on the testing data and displays the accuracy.
  2.) nntrain(train_data,train_label) - Is a function in which we need to set the number of levels, number of nodes at each level and max iteration count manually. After that it creates the model with intial weight vectors and calls the buildmodel.m which updates the model.
  3.) activation(layer_id,input_vector,weights) - It computes the sigmoidal function of the given inputs.
  4.) buildmodel(model,iter_count,train_data,train_label) - It performs the forward pass and backward pass action, also computes the error value and exits if it reaches the threshold. Additionally for evaluation purposes it saves the iteration number, error value and accuracy in the corresponding iteration onto a file name temp.mat.
  5.) nnpredict(model,test_data) - It performs the forward pass on the given data and then returns the labels for the given test_data.

In order to execute the program please remember to set the values for number of levels, number of nodes in each level, iteration count and error threshold if you want to change these parameters. After updating the corresponding values run loaddata script.