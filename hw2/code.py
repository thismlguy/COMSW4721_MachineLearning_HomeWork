#Python3
import numpy as np
import csv
import matplotlib.pylab as plt
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

#constants:
test_using_sklearn = True

def read_file(filename):
	data = []
	with open('hw2-data/%s'%filename) as file:
		content = csv.reader(file)
		for c in content:
			data.append(c)
	return np.matrix(data).astype(np.float32)

def naive_bayes(X_train,Y_train,X_test,Y_test):

    #get ratio of 1s:
    pi = np.mean(Y_train)
    # print(pi)

    X_train2 = X_train.copy()
    X_train2[:,-3:] = np.apply_along_axis(np.log,0,X_train2[:,-3:])
    # print(X_train2[(0,1,2),-3:]) 
    # print(np.apply_along_axis(np.log,0,X_train2[(0,1,2),-3:]))
    # print(np.apply_along_axis(np.mean,0,X_train2).shape)
    
    zero_vals = np.array([x==0 for x in Y_train.reshape(1,-1).tolist()[0] ])

    weights = [np.apply_along_axis(np.mean,0,
                                    X_train2[zero_vals,]),
               np.apply_along_axis(np.mean,0,
                                    X_train2[np.invert(zero_vals),])
               ]
    #invert the last 3 weights because MLE is inverse
    weights[0][-3:] = 1/weights[0][-3:]
    weights[1][-3:] = 1/weights[1][-3:]

    def predict_bern(row,coef,pi):
        #initialize to class priors
        result=[1-pi,pi]
        row=[item for sublist in row.flatten().tolist() for item in sublist]
        for j in range(2):
            #mutliply weights by row for bern
            # print(row.flatten())
            
            # print(j)
            # print(row)
            tr_bern = [(coef[j][i]**row[i])*((1-coef[j][i])**(1-row[i])) 
                            for i in range(54)]
            #mutliply weights by row for pareto
            tr_pareto = [coef[j][i]*(row[i]**(-coef[j][i]-1)) 
                            for i in range(54,57)]
            result[j]*=np.prod(tr_bern)*np.prod(tr_pareto)
        return np.argmax(result)

    def crosstab(list1,list2):
        # print(list1.shape)
        # print(list2.shape)
        print('predict\tactual\tnum')
        for l in [0.0,1.0]:
            for m in [0.0,1.0]:
                print('%d\t%d\t%d'%(l,m,
            sum( [(list1[0,i]==l)&(list2[0,i]==m) for i in range(list1.shape[1])]
                          )))

    #results for train:
    # predict_train = np.apply_along_axis(predict_bern,1,X_train,coef=weights,pi=pi).flatten().reshape(-1,1)
    # acc_train = sum(predict_train==Y_train)/len(Y_train)
    # print('train accuracy: %f'%acc_train)
    # print('train crosstab:')
    # crosstab(predict_train.reshape(1,-1),
    #          Y_train.reshape(1,-1))


    #results for test:
    predict_test = np.apply_along_axis(predict_bern,1,X_test,coef=weights,pi=pi).flatten().reshape(-1,1)
    acc_test = sum(predict_test==Y_test)/len(Y_test)
    print('\ntest accuracy: %f'%acc_test)
    print('test crosstab:')
    crosstab(predict_test.reshape(1,-1),
             Y_test.reshape(1,-1))

    #stem plot:
    fig,ax = plt.subplots(2,1)
    ax1,ax2=ax.ravel()
    markerline, stemlines, baseline = ax1.stem(range(1,55), weights[0][:54], '-.')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color','r', 'linewidth', 2)
    ax1.set_title('Weights for y=0')

    markerline, stemlines, baseline = ax2.stem(range(1,55), weights[1][:54], '-.')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color','r', 'linewidth', 2)
    ax2.set_title('Weights for y=1')

    plt.tight_layout()
    plt.xlabel('Dimension')
    plt.ylabel('Estimated Weights')
    plt.savefig('hw2_naive_bayes.png')
    plt.show()


def KNN(X_train,Y_train,X_test,Y_test):
    #standardize last 3 columns:
    X_train2 = X_train.copy()
    X_test2 = X_test.copy()
    Y_train2 = np.squeeze(np.asarray(Y_train))
    Y_test2 = np.squeeze(np.asarray(Y_test))

    # for i in range(54,57):
    #     xmean = np.mean(X_train2[:,i])
    #     xstd = np.std(X_train2[:,i])
    #     X_train2[:,i] = (X_train2[:,i]-xmean)/xstd
    #     X_test2[:,i] = (X_test2[:,i]-xmean)/xstd

    def get_abs_dist(list1,list2):
        # print(list1.shape)
        # print(.shape)
        list2 = np.squeeze(np.asarray(list2))
        # print(list1-list2)
        return np.sum(np.abs(list1-list2))

    knn_outcome = []
    #make a matrix of outcomes from top 20 sorted distances:
    for i in range(X_test2.shape[0]):

        #calculate all distances:
        dist = np.apply_along_axis(get_abs_dist,1,X_train2,list2=X_test2[i,:])
        #find top 20 distances:
        topk_ind = np.argsort(dist)[:20]
        #store the outcome of those 20:
        knn_outcome.append(Y_train2[topk_ind])
    # print(knn_outcome)

    accuracy_scores = [] 
    k_values = range(1,21)
    for k in k_values:
        
        #get predictions for the k-value:
        predict_test = np.apply_along_axis(
                                    lambda x: int(np.mean(x)>=0.5), 
                                    1, 
                                    [x[:k] for x in knn_outcome])
        
        accuracy = sum(np.array(predict_test)==Y_test2)/len(predict_test)
        accuracy_scores.append(accuracy)

    plt.figure()
    plt.plot(k_values, accuracy_scores)
    plt.xticks(k_values)
    plt.xlabel('K-values')
    plt.ylabel('Accuracy scores')
    plt.title('Accuracy scores vs K (KNN)')
    plt.savefig('hw2_knn.png')
    plt.show()

def logistic_regression(X_train,Y_train,X_test,Y_test,
                        implement_regular=False, implement_newton=False):
    X_train2 = X_train.copy()
    X_test2 = X_test.copy()
    Y_train2 = Y_train.copy()
    Y_test2 = Y_test.copy()

    #standardize data:  NOT ALLOWED!!
    # for i in range(54,57):
    #     xmean = np.mean(X_train2[:,i])
    #     xstd = np.std(X_train2[:,i])
    #     X_train2[:,i] = (X_train2[:,i]-xmean)/xstd
    #     X_test2[:,i] = (X_test2[:,i]-xmean)/xstd

    #Add a column of 1s to the data:
    X_train2 = np.column_stack((X_train2,np.ones(X_train2.shape[0])))
    X_test2 = np.column_stack((X_test2,np.ones(X_test2.shape[0])))

    #make 0 to -1:
    Y_train2[Y_train2==0]=-1
    Y_test2[Y_test2==0]=-1

    #initialize weights:
    weights = np.zeros(X_train2.shape[1]).reshape(-1,1)

    #define a sigmoit function:
    def sigmoid(x):
        print(x)
        return 1/(1+np.exp(-x))

    if implement_regular:
        iterations=100#00
        objective = []
        for t in range(1,iterations+1):

            # print('t=%d'%t)
            eta = 1/(1e5*np.sqrt(t+1))

            sigm_i = expit(np.multiply(Y_train2,X_train2.dot(weights)))
            # print(sigm_i.shape)

            #objective for previous iteration:
            objective.append(np.sum(np.log(sigm_i+1e-10)))

            #determine the constant for each i
            update = X_train2.T.dot(np.multiply(Y_train2,1-sigm_i))
            # print(update.shape)

            weights += eta*update

            # print(weights[:10])
        figtitle = 'Logistic Regression (Steepest Ascnet)'
        figsave = 'hw2_logreg.png'

    if implement_newton:
        iterations=100
        objective = []
        for t in range(1,iterations+1):
            eta = 1/np.sqrt(t+1)

            sigm_i = expit(np.multiply(Y_train2,X_train2.dot(weights)))
            # print(sigm_i.shape)

            #objective for previous iteration:
            objective.append(np.sum(np.log(sigm_i+1e-10)))

            #determine the second order gradient:
            # second_grad = -np.multiply(
            #                     np.multiply(sigm_i,1-sigm_i),
            #                     X_train2.dot(X_train2.T)
            #                     )
            second_grad = -np.multiply(
                                np.multiply(sigm_i,1-sigm_i),
                                X_train2).T.dot(X_train2)
            # print(second_grad)
            #determine the constant for each i
            first_grad = X_train2.T.dot(np.multiply(Y_train2,1-sigm_i))
            # print(update.shape)

            weights -= eta*np.linalg.inv(second_grad).dot(first_grad)

        figtitle = 'Logistic Regression (Newton Method)'
        figsave = 'hw2_newton.png'

    print(weights)

    predict_test = np.sign(X_test2.dot(weights))
    accuracy = sum(predict_test==Y_test2)/len(predict_test)
    print("Test set accuracy:",accuracy)


    # print(objective[:10])
    # plt.figure()
    # plt.plot(range(1,iterations+1),objective)
    # plt.xlabel('iterations')
    # plt.ylabel('objective function')
    # plt.title(figtitle)
    # plt.savefig(figsave)
    # plt.show()


def main():
    args = sys.argv

    X_train = read_file('X_train.csv')
    X_test = read_file('X_test.csv')
    Y_train = read_file('y_train.csv')
    Y_test = read_file('y_test.csv')
    assert np.shape(X_train)==(4508,57)
    assert np.shape(X_test)==(93,57)
    assert np.shape(Y_train)==(4508,1)
    assert np.shape(Y_test)==(93,1)

    #test on subsets:
    # X_train = X_train[,:]
    # Y_train = Y_train[:100]

    #call naive bayes
    if 'naive' in args:
        naive_bayes(X_train,Y_train,X_test,Y_test)

    #KNN:
    if 'knn' in args:
        KNN(X_train,Y_train,X_test,Y_test)

    #logistic:
    if 'logreg1' in args:
        logistic_regression(X_train,Y_train,X_test,Y_test,
                            implement_regular=True)
    if 'logreg2' in args:
        logistic_regression(X_train,Y_train,X_test,Y_test,
                            implement_newton=True)

if __name__ == "__main__":
	main()