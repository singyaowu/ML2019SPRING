import numpy as np
import sys
sigmoid = lambda s: (1.0 / (1 + np.exp(-s)))

#bash ./hw2_logistic.sh train.csv test.csv X_train Y_train X_test prediction.csv
if __name__ == "__main__":
    # read training data
    raw_x = np.genfromtxt(sys.argv[1], delimiter=',', dtype=np.float64)
    raw_y = np.genfromtxt(sys.argv[2], delimiter=',', dtype=np.float64)
    
    x = raw_x[1:,:]
    y = raw_y[1:,np.newaxis]

    print('read input finish')
    
    # extract feature
    title_fp = open(sys.argv[1], 'r')
    title = title_fp.readline().split(',')
    title_fp.close()
    
    pick_dict = dict(zip(title, range(len(title))))
    # age,fnlwgt,sex,capital_gain,capital_loss,hours_per_week,
    # 
    # workclass: Federal-gov, Local-gov, Never-worked, Private, Self-emp-inc, Self-emp-not-inc, State-gov, Without-pay,?_workclass,
    # 
    # educatuin: 10th, 11th, 12th, 1st-4th, 5th-6th, 7th-8th, 9th, 
    #            Assoc-acdm, Assoc-voc, Bachelors, Doctorate, HS-grad, Masters, Preschool, Prof-school, Some-college,
    # 
    # marital_status: Divorced, Married-AF-spouse, Married-civ-spouse, Married-spouse-absent, Never-married, Separated, Widowed,
    # 
    # occupation: Adm-clerical, Armed-Forces, Craft-repair, Exec-managerial, Farming-fishing, Handlers-cleaners, Machine-op-inspct,
    #             Other-service, Priv-house-serv, Prof-specialty, Protective-serv, Sales, Tech-support, Transport-moving,?_occupation,
    # 
    # relationship: Husband, Not-in-family, Other-relative, Own-child, Unmarried, Wife, 
    # 
    # native country: Amer-Indian-Eskimo, Asian-Pac-Islander, Black, Other, White, Cambodia, Canada, China, Columbia, Cuba,
    #                 Dominican-Republic, Ecuador, El-Salvador, England, France, Germany, Greece, Guatemala, Haiti, Holand-Netherlands,
    #                 Honduras, Hong, Hungary, India, Iran, Ireland, Italy, Jamaica, Japan, Laos, Mexico, Nicaragua, Outlying-US(Guam-USVI-etc),
    #                 Peru, Philippines, Poland, Portugal, Puerto-Rico, Scotland, South, Taiwan, Thailand, Trinadad&Tobago, United-States, Vietnam,
    #                 Yugoslavia,?_native_country
    
    second_id = ['age','fnlwgt','sex', 'capital_gain','capital_loss','hours_per_week']
    third_id = ['age','sex', 'capital_gain','capital_loss','hours_per_week']
    
    selected = [pick_dict[id_name] for id_name in second_id]
    second_feature = [ ( x[:, id:id+1 ] ) for id in selected]
    second_feature = np.concatenate( tuple(second_feature), axis=1)

    selected2 = [pick_dict[id_name] for id_name in third_id]
    third_feature = [ ( x[:, id:id+1 ] ) for id in selected2]
    third_feature = np.concatenate( tuple(third_feature), axis=1)
    print(second_feature.shape, x.shape)
    #x = np.delete(x, del_ids, axis=1)
    x = np.concatenate((x, second_feature**2, third_feature**3), axis=1)
    num_data, dim = x.shape
    #normalization
    mean = np.mean(x,axis=0)
    std = np.std(x, axis=0)
    for i in range(dim):
        x[:,i] = (x[:,i] - mean[i] )/(1 if std[i] == 0 else std[i])
    x = np.concatenate((np.ones(shape=(num_data,1)), x), axis=1).astype(np.float64)
    dim += 1
    
    cut = int(4/4 * num_data)
    val_x = x[cut:, :]
    val_y = y[cut:, :]
    x = x[:cut,:]
    y = y[:cut,:]
    num_data, dim = x.shape
    
    # training
    
    w = 0.1 * np.ones(shape= (dim, 1), dtype=np.float64)
    lr = 0.1
    ld = 0.1
    iteration = 3000
    sum_grad = np.zeros(shape=(dim,1),dtype=np.float64)
    for i in range(iteration):
        f_x = sigmoid(x.dot(w))
        if i % 1000 == 0:            
            loss = -np.sum(y * (np.log(f_x + 1e-8)) + (1.0 - y) * np.log(1.0 - f_x + 1e-8))
            y_hat = np.around(f_x +1e-8).reshape(-1).astype(int)
            y_test = y.reshape(-1).astype(int)
            accuracy = np.sum([ y1 == y2 for y1, y2 in zip(y_test, y_hat)]) / num_data
            print('iteration:',i,'loss:', loss, 'accuracy:', accuracy)
        
        gradient = np.zeros(shape=(dim, 1), dtype=np.float64)
        gradient = -x.T.dot(y - f_x) + 2 * ld * w
        sum_grad += gradient**2
        
        w -= lr * gradient / (np.sqrt(sum_grad + 1e-8))
    
    if val_y.shape[0] > 0:
        expect_y = np.around(sigmoid(val_x.dot(w))).astype(int)
        val_accuracy = sum([ int(val_y[i][0]) == expect_y[i][0] for i in range(val_y.shape[0])]) / val_y.shape[0]
        print('validation:', val_accuracy)


    np.save('weight_best.npy', w)
    np.save('mean_best.npy', mean)
    np.save('std_best.npy', std)
    np.save('selected.npy', np.array(selected))
    np.save('selected2.npy', np.array(selected2))