function [gain,label,set] = selectAttribute(r_set, b_ids, data, labels, p_labels,o_id,d)

% Computes the information gain on label probability for each feature in data
% data: d x n matrix of d features and n examples
% labels: n x 1 vector of class labels for n examples
% gain: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))

% REPLACE THE FOLLOWING WITH YOUR IMPLEMENTATION
    entropy_data=calculateEntropy(p_labels);
    set=[];
    if(entropy_data==0)
        gain=-1;label=find(p_labels==1);
        return;
    end
    sets.zero=sparse(size(b_ids, 1),1);
    sets.zero.p_labels=sparse(size(p_labels,1),1);
    sets.one=sparse(size(b_ids, 1),1);
    sets.one.p_labels=sparse(size(p_labels,1),1);
    gain = sparse(size(b_ids, 1),1);
    for i = 1 : size(b_ids,1)
        [label0,s0,label1,s1]=getLabels(r_set,b_ids(i,1),data,labels);
        sets.zero(i,1).p_labels=label0;
        sets.one(i,1).p_labels=label1;
        gain(i,1)=(s0*calculateEntropy(label0)+s1*calculateEntropy(label1))/(s0+s1);
        split=calculateEntropy([s0;s1]/(s0+s1));
        gain(i,1)=gain(i,1)/split;
    end
    gain=entropy_data-gain;
    if(o_id==1)
        [gain,set]=sort(gain);
        set=set(1:d,1);
        gain=-1;label=-1;
        return;
    end
    [gain,label]=max(gain);
    set.zero=sets.zero(label,1).p_labels;
    set.one=sets.one(label,1).p_labels;
end
function entropy = calculateEntropy(labels)
    entropy=0;
    label=find(labels==1);
    if(size(label,1)==1)
        return;
    end
    for i = 1 : size(labels,1)
        if(labels(i)~=0)
            entropy=entropy+abs(labels(i)*log2(labels(i)));
        end
    end
end
function [label0,s0,label1,s1] = getLabels(r_set, b_id, data, labels)
    r_ids_1=r_set(find(data(r_set,b_id)),1);
    r_ids_0=setdiff(r_set,r_ids_1);
    label0=sparse(size(unique(labels),1),1);
    label1=sparse(size(unique(labels),1),1);
    for i = 1 : size(r_ids_0,1)
        label0(labels(r_ids_0(i,1)))=label0(labels(r_ids_0(i,1)))+1;
    end
    for i = 1 : size(r_ids_1,1)
        label1(labels(r_ids_1(i,1)))=label1(labels(r_ids_1(i,1)))+1;
    end
    s0=size(r_ids_0,1);
    s1=size(r_ids_1,1);
    if(s0~=0)
        label0=label0/s0;
    else
        label0(:,1)=0;
    end
    if(s1~=0)
        label1=label1/s1;
    else
        label1(:,1)=0;
    end
end
