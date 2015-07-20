function cifar_eval_hamming2( trn_list, trn_label, trn_binary, tst_list, tst_label, tst_binary, bits)   
K = 1000;
QueryTimes = 10000;
fname = sprintf('log_cifar10_hamming2%d.txt',bits);%Hamming radius <=2
fid = fopen(fname, 'wt');

correct = 0;
myPrecision = 0;

for i = 1:QueryTimes
    
    img_path = tst_list(i,1);
    query_label = get_label(img_path, tst_label);
    fprintf('query %d\n',i);
    query_binary = tst_binary(:,i);
    
    tic
    similarity = pdist2(trn_binary',query_binary','hamming');
    %similarity = pdist2(trn_binary',query_binary','euclidean');
    toc
    fprintf('Complete Query [Hamming] %.2f seconds\n',toc);

    [x2,y2]=sort(similarity);
    
    Threshold = 2;
    selected_range = (find(x2<(Threshold/bits)));
    s = max(selected_range);
    %new_set = y(1:s);
    
    correct = 0;
    for j = 1:s
        filename = trn_list(y2(j),1);
        retrieval_label = get_label(filename,trn_label);
        
        if (query_label==retrieval_label)
            correct = correct + 1;
        end
    end
    if s>0
    myPrecision = myPrecision + correct/s;
    end
end    

  
fprintf(fid, '%f\n',myPrecision);

fclose(fid);
  
     
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

