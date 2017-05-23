#Python3
import numpy as np
import pandas as pd
import csv
import matplotlib.pylab as plt
import sys

def read_file(filename):
    data = []
    with open(filename) as file:
        content = csv.reader(file)
        for c in content:
            data.append(c)
    return np.matrix(data).astype(np.float32)

#############################
#### PART A - KMEANS
#############################
class kmeans_gaussian(object):
    
    def __init__(self, ntrain, mix_weights):
        self.ntrain = ntrain
        self.mix_weights = mix_weights
        self.generate_data()
    
    def generate_data(self):
        cov = np.matrix([[1,0],[0,1]])
        mean1 = np.array([0,0])
        mean2 = np.array([3,0])
        mean3 = np.array([0,3])
        gauss1 = np.random.multivariate_normal(mean1,cov,self.ntrain)
        gauss2 = np.random.multivariate_normal(mean2,cov,self.ntrain)
        gauss3 = np.random.multivariate_normal(mean3,cov,self.ntrain)
        # print(gauss1.shape,gauss2.shape,gauss3.shape)

        #generate random draws:
        choice = np.random.choice(range(3), 
                                  size=500, p=self.mix_weights)
        # print(choice)
        self.data = np.concatenate(( gauss1[choice==0,:],
                                     gauss2[choice==1,:],
                                     gauss3[choice==2,:] ))

        # print(gauss1[:4,:])
        # print(gauss2[:4,:])
        # print(gauss3[:4,:])
        # print(self.data.shape)

    def set_k(self,k):
        self.k=k

    def initialize_cluster_centers(self):
        self.centers = np.random.uniform(low=0,high=1,size=(self.k,2))
        self.objective = []
        # print(self.centers)

    def get_closest_cluster(self, row):
        # print(self.centers - row)
        # print((self.centers - row)**2)
        # print(np.argmin(np.sum((self.centers - row)**2,axis=1)))
        errors = np.sum((self.centers - row)**2,axis=1)
        sel = np.argmin(errors)
        return (sel,errors[sel])

    def train(self, T):
        # print(self.centers)
        # print(self.data[:2])
        for t in range(T):
            self.cluster_assgn = np.apply_along_axis(self.get_closest_cluster,
                                                  1,self.data)
            # print(cluster_assgn)
            self.objective.append(np.sum(self.cluster_assgn[:,1]))

            #update centers:
            for i in range(self.k):
                self.centers[i,:] = np.mean(self.data[self.cluster_assgn[:,0]==i],axis=0)


def line_plot(x,y1,y2,xticks,xlabel,ylabel,title,savfigname,leg=None):
    plt.figure()
    plt.plot(x,y1,'b')

    if y2 is not None:
        plt.plot(x,y2,'g')
        plt.legend(leg)
    plt.xticks(xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savfigname)
    plt.show()

def kmeans():

    T=20
    n_train = 500
    mix_weights = (0.2,0.5,0.3)

    #create a class object:
    km = kmeans_gaussian(n_train,mix_weights)

    # set k-values
    kvalues = range(2,6)
    colors = ['blue','green','red','black','yellow']

    #store the cluster assignments:
    k_backup = [3,5]
    cluster_assgn35 = []

    plt.figure()
    for i in range(len(kvalues)):
        km.set_k(kvalues[i])
        km.initialize_cluster_centers()
        km.train(T)

        plt.plot(range(1,T+1),km.objective,colors[i])

        #store cluster assignments for k=3,5
        if kvalues[i] in k_backup:
            cluster_assgn35.append(km.cluster_assgn[:,0])

    plt.xticks(range(1,T+1))
    plt.xlabel('Iterations')
    plt.ylabel('Objective')
    plt.title('Objective vs Iteration for K = [2,3,4,5]')
    plt.legend(['K = %d'%i for i in kvalues])
    # plt.savefig('hw4_1a_kmean_obj')
    # plt.show()

    #plot part b:
    for i in range(2):
        plt.figure()
        colors_arr = [colors[int(x)] for x in cluster_assgn35[i]]
        plt.scatter(km.data[:,0],km.data[:,1],c=colors_arr)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Scatter plot with cluster assignment for K=%d'%k_backup[i])
        plt.savefig('hw4_2_k%d.png'%k_backup[i])
        plt.show()

#############################
#### PART B - MATRIX FACT
#############################

class matrix_factorization(object):

    def __init__(self, train_file, test_file, metadata_file,
                 nusers=943, nmovies=1682,
                 var=0.25, d=10, lmb=1):
        self.nusers = nusers
        self.nmovies = nmovies

        self.var = var
        self.d = int(d)
        self.lmb= int(lmb)

        self.load_data(train_file, test_file)
        self.load_metadata(metadata_file)

    def load_metadata(self,file):
        with open(file) as f:
            self.movies = np.array([x.rstrip('\n') for x in f.readlines()])
        # print(self.movies.shape)
        # print(len(self.movies))
        # print(self.movies[:5])

    def create_matrix_from_file(self,file,train=False,test=False):
        #declare M:
        matrix = np.repeat(np.nan,self.nusers*self.nmovies).reshape(self.nusers, self.nmovies)
        # print(self.M.shape)

        #load info into it:
        with open(file) as f:
            for row in f:
                val = row.rstrip('\n').split(',')
                # print(val)
                matrix[int(val[0])-1, int(val[1])-1] = float(val[2])
                if train:
                    self.num_train_cases+=1
                if test:
                    self.num_test_cases+=1
        return matrix

    def load_data(self,train_file, test_file):
        
        self.num_train_cases = 0
        self.num_test_cases = 0

        self.M = self.create_matrix_from_file(train_file, train=True)
        self.Mtest = self.create_matrix_from_file(test_file, test=True)        
        # print(self.M.shape, self.Mtest.shape)

    def generate_data(self):
        self.Q = np.repeat(np.nan,self.nusers*self.d).reshape(self.nusers, self.d)
        self.R = np.random.multivariate_normal(np.repeat(0,self.d),
                                               np.identity(self.d)/self.lmb,
                                               self.nmovies).T
        # print(self.Q.shape, self.R.shape)

    def calc_error(self, matrix):

        predicted = self.Q.dot(self.R)
        observed_ind = ~np.isnan(matrix)

        error = ((matrix[observed_ind] - predicted[observed_ind])**2).sum()
        return error

    def calc_sqsum(self,matrix):
        return (matrix**2).sum()

    def run_algorithm(self):

        self.objective = []
        for p in range(100):
            for i in range(self.nusers):
                observed_ind = ~np.isnan(self.M[i,:])
                # print(sum(observed_ind))
                Ri = self.R[:,observed_ind]
                Mi = self.M[i,observed_ind]
                # print(Ri.shape, Mi.shape)
                self.Q[i,:] = np.linalg.inv( Ri.dot(Ri.T) + 
                                        self.lmb*self.var*np.identity(self.d)).dot(Ri.dot(Mi.T))

            for j in range(self.nmovies):
                observed_ind = ~np.isnan(self.M[:,j])
                # print(sum(observed_ind))
                Qj = self.Q[observed_ind,:]
                Mj = self.M[observed_ind,j]
                # print(Ri.shape, Mi.shape)
                self.R[:,j] = np.linalg.inv(Qj.T.dot(Qj)+self.lmb*self.var*np.identity(self.d)).dot(Qj.T.dot(Mj.T))

            # print(np.isnan(self.Q).sum(),np.isnan(self.R).sum())
            # print(self.calc_sqsum(self.Q),self.calc_sqsum(self.R),self.calc_error(self.M))
            if p>1:
                obj_neg = ( self.calc_error(self.M)/(2*self.var) + 
                           self.calc_sqsum(self.Q)*self.lmb/2 + 
                           self.calc_sqsum(self.R)*self.lmb/2 )
                self.objective.append(-obj_neg)
            # print(self.obj_neg)

    def find_map(self):

        num_runs = 10
        x_vals = list(range(2,100))

        #initialize table
        results = pd.DataFrame(index=range(num_runs),
                               columns=['s.no.','objective','test_rmse'])

        #get predictions for required movies:
        self.query_movies = ["Star Wars", "My Fair Lady", "GoodFellas"]
        self.movie_results = pd.DataFrame(index=range(10),
                                     columns=self.query_movies)
        self.dist_results = pd.DataFrame(index=range(10),
                                     columns=self.query_movies)

        #define minimum objective:
        max_obj = -np.inf
        results['s.no.'] = list(range(1,num_runs+1))

        plt.figure()
        for i in range(num_runs):
            self.generate_data()

            self.run_algorithm()

            plt.plot(x_vals, self.objective, label='run_%d'%(i+1))

            #get objective:
            results.loc[i,'objective'] = self.objective[-1]
            results.loc[i,'test_rmse'] = np.sqrt(self.calc_error(self.Mtest)/self.num_test_cases)

            #update movie pred if required:
            if self.objective[-1]>max_obj:
                max_obj = self.objective[-1]
                self.update_query_results()

        plt.xticks([int(x) for x in np.linspace(2,100,10)])
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        plt.title('Objective function for 10 runs')
        plt.legend(loc='best')
        plt.savefig('hw4_2a_obj.png')
        plt.show()

        #export objective:
        results = results.sort_values(by='objective',axis=0,ascending=False)
        print(results)
        results.to_csv('hw4_2a_obj_rmse.csv',index=False)

        #print query results
        print(self.movie_results)
        print(self.dist_results)

        self.movie_results.to_csv('hw4_2b_query_movie.csv',index=False)
        self.dist_results.to_csv('hw4_2b_query_distances.csv',index=False)
    
    def update_query_results(self):

        for movie in self.query_movies:
            movie_id = [i for i in range(self.nmovies) 
                        if movie in self.movies[i]][0]
            # print(movie_id)
            distances = np.sqrt(((self.R - self.R[:,movie_id].reshape(-1,1))**2).sum(axis=0))
            
            # print(distances.shape)
            min_movies_id = np.argsort(distances)[1:11]
            # print(min_movies_id)
            # print(self.movies[min_movies_id])

            self.movie_results[movie] = self.movies[min_movies_id]
            self.dist_results[movie] = distances[min_movies_id]
            # print(movie_id)
            # print(np.sort(distances))
            # print(min_movies_id)


def mfact(train_file, test_file, movie_map):

    mf = matrix_factorization(train_file, test_file, movie_map)
    mf.find_map()
    print(mf.R)


def main():
    args = sys.argv

    #call naive bayes
    if 'kmeans' in args:
        kmeans()

    if 'mf' in args:
        trainf = "data/ratings.csv"
        testf = "data/ratings_test.csv"
        mmap = "data/movies.txt"
        mfact(trainf, testf, mmap)



    

if __name__ == "__main__":
    main()