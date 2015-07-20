function cifar_eval( trn_list, trn_label, trn_binary, tst_list, tst_label, tst_binary, bits)   
K = 1000;
QueryTimes = 10000;
fname = sprintf('log_cifar10_%d.txt',bits);%P@K
fid = fopen(fname, 'wt');
fname_map = sprintf('log_cifar10_%d_MAP.txt',bits);%MAP 
fid_map = fopen(fname_map, 'wt');


correct = zeros(K,1);
total = zeros(K,1);
error = zeros(K,1);
AP = zeros(QueryTimes,1);

for i = 1:QueryTimes
    
    img_path = tst_list(i,1);
    query_label = get_label(img_path, tst_label);
    fprintf('query %d\n',i);
    query_binary = tst_binary(:,i);
    
    tic
    %similarity = pdist2(trn_binary',query_binary','hamming');
    similarity = pdist2(trn_binary',query_binary','euclidean');
    toc
    fprintf('Complete Query [Euclidean] %.2f seconds\n',toc);

    [x2,y2]=sort(similarity);
    
    
    buffer_yes = zeros(K,1);
    buffer_total = zeros(K,1);
    total_relevant = 0;
    
    for j = 1:K
        filename = trn_list(y2(j),1);
        retrieval_label = get_label(filename,trn_label);
        
        if (query_label==retrieval_label)
            buffer_yes(j,1) = 1;
            total_relevant = total_relevant + 1;
        end
        buffer_total(j,1) = 1;

    end

    for j = 1:K
        for kk = 1:j
            correct(j,1) = correct(j,1) + buffer_yes(kk,1);
            total(j,1) = total(j,1) + buffer_total(kk,1);
        end
    end
    
    for j = 1:K
        precision = correct(j,1)/total(j,1);
        AP(i,1) = AP(i,1) + precision*buffer_yes(j,1);
    end
    AP(i,1) = AP(i,1)/(total_relevant+0.00001);
    
end    
    accuracy = correct./total;
    plot(1:K,accuracy);



for i = 1:K
    fprintf(fid, '%d %f\n',i,correct(i,1)/total(i,1));
end    

MAP = 0;
for i = 1:QueryTimes
    MAP = MAP + AP(i,1);
end
%MAP = MAP/QueryTimes;
fprintf(fid_map, '%f\n', MAP);

fclose(fid);
fclose(fid_map);    
     
end


function [label_output] = get_label(img_path, labels)
    image_filename = regexp(img_path{1}, '/', 'split');
    str0 = image_filename(6);
    
    base = 0;
    if (strcmp(str0,'test')==1)
        base = 0;
    end    
    if (strcmp(str0,'batch1')==1)
        base = 0;
    end
    if (strcmp(str0,'batch2')==1)
        base = 10000;
    end
    if (strcmp(str0,'batch3')==1)
        base = 20000;
    end
    if (strcmp(str0,'batch4')==1)
        base = 30000;
    end
    if (strcmp(str0,'batch5')==1)
        base = 40000;
    end
    
    str1 = image_filename(7);
    str2 = str1{1}(1:end-4);%without '.jpg'
    x = str2num(str2);
    label_output = labels(x+base);
    
end

