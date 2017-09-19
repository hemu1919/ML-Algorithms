% script loads data from 20newsgroup_subset
train_data = load('../20newsgroup_subset/train6.data');
test_data = load('../20newsgroup_subset/test6.data');
train_label = load('../20newsgroup_subset/train6.label');
train_label(3388,1)=train_label(3387,1);
test_label = load('../20newsgroup_subset/test6.label');
params.alpha=1.0;
params.depth=10;
params.steps=0.01;
folds=10;
w_count=5000;
%a=[1 1 4;1 2 2;2 2 1;3 2 3;3 3 2;4 1 1;4 3 4;5 2 2;5 3 1];
%b=[1;2;3;2;3];
%c=sparse(a(:,1),a(:,2),a(:,3));
