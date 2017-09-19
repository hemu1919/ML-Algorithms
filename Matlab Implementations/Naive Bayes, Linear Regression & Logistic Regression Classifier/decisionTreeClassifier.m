function label = decisionTreeClassifier(testData,model)
    data=sparse(testData(:,1),testData(:,2),testData(:,3));
    docs=size(data,1);
    books=size(data,2);
    label=decisionTreePredict((1:docs)',(1:books)', data, model.Model, model.Labels);
end
