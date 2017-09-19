function label = bayesClassifier(testData,model)
    data=sparse(testData(:,1),testData(:,2),testData(:,3));
    docs=size(data,1);
    label=naiveBayesPredict((1:docs)', data, model.Model, model.Labels);
end
