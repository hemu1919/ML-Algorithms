function label = nnpredict(model, test_data)
    label=sparse(size(test_data,1),1);
    for i = 1 : size(test_data,1)
        a=struct('level1',[1 test_data(i,:)]);
        for j = 2 : model.levels
            level=getfield(model,strcat('level',int2str(j)));
            a1=[];
            for k = 1 : level.count
                a1(end+1)=activation(j,getfield(a,strcat('level',int2str(j-1))),getfield(level.weights,strcat('node',int2str(k))));
            end
            a=setfield(a,strcat('level',int2str(j)),[1 a1]);
        end
        temp=getfield(a,strcat('level',int2str(model.levels)));
        [~,ind]=max(temp(1,2:end));
        label(i)=model.labels(ind);
    end
end