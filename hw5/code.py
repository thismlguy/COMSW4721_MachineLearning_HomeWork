#Python3
import numpy as np
import pandas as pd
import csv
import matplotlib.pylab as plt
import sys
from scipy.sparse.linalg import eigs

def read_file(filename):
    data = []
    with open(filename) as file:
        content = csv.reader(file)
        for c in content:
            data.append(c)
    return np.matrix(data).astype(np.float32)

#############################
#### PART A - MARKOV CHAIN
#############################
class markov_chain(object):
    
    def __init__(self, size, file_m, file_names):
        
        self.n = 760
        #initialize M:
        self.M = np.zeros((self.n,self.n))

        #generate M from file:
        self.create_matrix_from_file(file_m)
        self.read_team_names(file_names)

        #get eigen value:
        self.get_eigen_vector()

    
    def create_matrix_from_file(self,file):
        
        #load info into it:
        with open(file) as f:
            for row in f:
                teamA, ptsA, teamB, ptsB = [int(x) for x in 
                                    row.rstrip('\n').split(',')]
                # print(teamA, ptsA, teamB, ptsB)

                a_weight = ptsA/(ptsA+ptsB)
                a_wins = int(ptsA > ptsB)

                i = teamA-1
                j = teamB-1
                self.M[i,i] += a_wins + a_weight
                self.M[j,i] += a_wins + a_weight
                self.M[j,j] += 1-a_wins + 1-a_weight
                self.M[i,j] += 1-a_wins + 1-a_weight

        # print(np.sum(self.M))
        # print(self.M)
        #normalize M:
        self.M = self.M / np.sum(self.M, axis=1).reshape(-1,1)
        # print(self.M)

        # print(np.sum(self.M,axis=1))

    def read_team_names(self,file):
        
        #load info into it:
        with open(file) as f:
            self.team_names = np.array([x.rstrip('\n') 
                                       for x in f.readlines()])
    
    def get_eigen_vector(self):
        # w,v = np.linalg.eig(self.M)
        # print(max(w),w)
        # max_ind = np.where(w==max(w))[0][0]
        # print(max_ind) 
        # self.w_inf = v[:,max_ind]
        # rearrangedEvalsVecs = sorted(zip(evals,evecs.T),\
        #                             key=lambda x: x[0].real, reverse=True)
        self.w_inf = eigs(self.M.T,1)[1].flatten()
        #make sum 1:
        # print(self.w_inf[:5])
        self.w_inf = self.w_inf / np.sum(self.w_inf)
        print(self.w_inf[:5])
        # print(self.w_inf.shape)
        

    def train(self, T):
        
        #initialize uniform:
        # w = np.zeros(self.n)
        # w[np.random.random_integers(0,self.n-1,1)] = 1
        w = np.repeat(1/760, 760)

        #store diff:
        self.diff = []
        t_ranks = np.logspace(1,4,4)
        self.results = pd.DataFrame(index=range(25),
                               columns=['ranks']+["t_%d"%i for i in t_ranks])
        self.results['ranks'] = list(range(1,26))
        #run iterations:
        for t in range(T):
            w = np.dot(w,self.M)
            # print((w-self.w_inf).shape)
            # w = w/np.sum(w)
            # print(sum(w))
            self.diff.append(np.sum(np.abs(w-self.w_inf)))

            if (t+1) in t_ranks:
                word_idx = np.argsort(w)[::-1][:25]
                self.results["t_%d"%(t+1)] = list(zip(
                        self.team_names[word_idx],
                        [format(x, '.3f') for x in w[word_idx]]))


                self.team_names[np.argsort(w)[::-1][:25]]
            


def mchain():

    T=10000
    n_teams = 760

    #create a class object:
    mc = markov_chain(n_teams, 'data/CFB2016_scores.csv', 'data/TeamNames.txt')
    mc.train(T)

    #plot diff variation:
    plt.figure()
    plt.plot(range(1,T+1),mc.diff)

    plt.xticks(np.linspace(1,T,9))
    plt.xlabel('Iterations')
    plt.ylabel('1-norm between w_inf and w_t')
    plt.title('Variation of difference between w_inf and w_t with iterations')
    plt.savefig('hw5_1b.png')
    plt.show()

    #export results:

    mc.results.to_csv('hw5_1a.csv',index=False)

#############################
#### PART B - NMF
#############################

class NMF(object):

    def __init__(self, file_doc, file_vocab, 
                 d=25, ndoc=8447, nvocab=3012):
        
        self.ndoc = ndoc
        self.nvocab = nvocab
        self.d=d
        
        #initialize doc file:
        self.X = np.zeros((nvocab,ndoc))
        self.create_matrix_from_file(file_doc)
        self.load_words(file_vocab)

        #initialize W, H:
        self.W = np.random.uniform(1,2,(self.nvocab,self.d))
        self.H = np.random.uniform(1,2,(self.d,self.ndoc))

        # print(self.W)

    
    def create_matrix_from_file(self,file):
        # load info into it:
        dinc=0
        with open(file) as f:
            for row in f:
                wordcounts = row.rstrip('\n').split(',')
                # print(val)
                for wc in wordcounts:
                    ind,cnt = [int(x) for x in wc.split(':')]
                    self.X[ind-1,dinc] = cnt
                dinc+=1
        # print(self.X)

    def load_words(self,file):
        
        with open(file) as f:
            self.words = np.array([x.rstrip('\n') 
                                       for x in f.readlines()])
    
    def train(self,T):

        self.objective = []

        WH = self.W.dot(self.H)
        A = self.X/(WH+1e-16)
        
        for i in range(T):
            if i%10==0:
                print(i)

            # print(np.multiply(self.H, self.W.T.dot(A)).shape)
            self.H = np.multiply(self.H, self.W.T.dot(A))/np.sum(self.W,axis=0).reshape(self.d,1)
            WH = self.W.dot(self.H)
            A = self.X/(WH+1e-16)
            self.W = np.multiply(self.W, A.dot(self.H.T))/np.sum(self.H,axis=1).reshape(1,self.d)

            # print(np.sum(self.H,axis=0))
            # print(np.sum(self.W,axis=1))
            WH = self.W.dot(self.H)
            A = self.X/(WH+1e-16)

            obj = np.sum(np.multiply(np.log(1/(WH+1e-16)),self.X) + WH)
            self.objective.append(obj)

    def get_words(self):
        #normalize W:
        self.W = self.W / (np.sum(self.W, axis=0).reshape(1,-1))
        word_idx = np.apply_along_axis(lambda x: np.argsort(x)[-10:][::-1],
                                  axis=0,arr=self.W)
        results = pd.DataFrame(index=range(10),
                              columns=['Topic_%d'%i for i in range(1,26)])
        # resuts = pd.DataFrame(index=range(5),
        #                       columns=range(5))
        # resuts['Topic'] = list(range(1,26))
        for i in range(25):
            # print(self.W[word_idx[:,i],i])
            results.iloc[:,i] = list(zip([format(x, '.3f') for x in 
                                                    self.W[word_idx[:,i],i]],
                                        self.words[word_idx[:,i]]))
        
        print(results)
        results.to_csv('hw5_2b_words.csv',index=False)



def nmf(fdoc, fvocab):
    T = 100

    nmf = NMF(fdoc, fvocab)
    nmf.train(T)
    nmf.get_words()
    # print(mf.R)

    plt.figure()
    plt.plot(range(1,T+1),nmf.objective)
    plt.xticks(np.linspace(1,T,10))
    plt.xlabel('Iterations')
    plt.ylabel('Objective')
    plt.title('Variation of objective with iterations')
    plt.savefig('hw5_2a.png')
    plt.show()


def main():
    args = sys.argv

    #call naive bayes
    if 'mchain' in args:
        mchain()

    if 'nmf' in args:
        trainf = "data/nyt_data.txt"
        testf = "data/nyt_vocab.dat"
        nmf(trainf, testf)


if __name__ == "__main__":
    main()