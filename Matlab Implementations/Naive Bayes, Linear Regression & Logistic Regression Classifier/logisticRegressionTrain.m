function [model,p_labels] = logisticRegressionTrain(r_set, trainData, trainLabels, params)
    p_labels='';
    books=size(trainData,2);
    label=unique(trainLabels);
    model=sparse(size(label,1),books+1);
    for  i = 1 : size(model,1)
        model(i,:)=getModel(r_set,label(i),trainData,trainLabels,params);
    end
end
function theta_max = getModel(r_set, l_id, trainData, trainLabels, params)
    docs=size(r_set,1);
    books=size(trainData,2);
    theta=sparse(books+1,1);
    j=0;j_min=intmax('int8');
    while(j<=500)
        j_curr=0;j_der=0;
        for i = 1 : docs
            h=sigmf(theta'*[1;trainData(r_set(i,1),1:books)'],[1 0]);
            y=trainLabels(r_set(i,1))==l_id;
            if(y==1)
                j_curr=j_curr+y*log(h);
            else
                j_curr=j_curr+(1-y)*log(1-h);
            end
            j_der=j_der+(h-y)*[1;trainData(r_set(i,1),(1:books))'];
        end
        j_curr=j_curr/-docs;
        if(j==0)
            j_min=j_curr;
            theta_max=theta;
        else
            [j_min,j_curr,j_min-j_curr]
            if((j_min-j_curr)<=0.00001 || j_curr>=j_min)
                break;
            elseif(j_curr<j_min)
                j_min=j_curr;
                theta_max=theta;
            end
        end
        j=j+1;
        j_der=j_der/docs;
        theta=theta-(params*j_der);
    end
end