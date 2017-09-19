clear all,clc
data=load('../optdigits_training.csv');
train_label=data(:,end);
train_data=data(:,1:end-1);
data=load('../optdigits_test.csv');
test_label=data(:,end);
test_data=data(:,1:end-1);
%set=getsubset(train_data,train_label);
%model=nntrain(set.data,set.label);
%label=nnpredict(model,set.data);
model=nntrain(train_data,train_label);
label=nnpredict(model,test_data);
test_accur=sum(label==test_label)/size(test_label,1);
display(test_accur);