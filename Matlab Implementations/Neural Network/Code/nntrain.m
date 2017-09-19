function model = nntrain(train_data, train_label)
    levels=3;
    %th=0.5;
    iter=100;
    %model=struct('levels',levels,'thresh',th,'labels',unique(train_label)');
    model=struct('levels',levels,'labels',unique(train_label)');
    nodes=[size(train_data,2) 30 size(unique(train_label),1)];
    step=[0.008 0.1];
    for i = 2 : levels
        level=struct('count',0,'weights',struct());
        level.count=nodes(i);
        level.step=step(i-1);
        for j = 1 : nodes(i)
            level.weights=setfield(level.weights,strcat('node',int2str(j)),rand(1,1+nodes(i-1)));
        end
        model=setfield(model,strcat('level',int2str(i)),level);
    end
    model=buildmodel(model,iter,train_data,train_label);
end