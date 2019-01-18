import os
import multiprocessing as mp
import sklearn.gaussian_process as gp
from BatchTools import *
from PlotTools import *
from BayesOptTools import *


class BayesOpt(Batch, Plot, BayesOptTools):

    def __init__(self, dimension=2, bounds=[[0, 10], [0, 10]], name_acq='ei', kernel='rbf',
                 nsteps=10, target=None, const=False):

        self.dim = dimension
        self.bounds = bounds
        self.find_max = False
        self.init_values = 4
        self.nb_procs = 4
        self.batch_size = 4
        self.xi = 0

        self.epsilon = 0.001

        if name_acq == 'ei':
            self.acq_function = self.expected_improvement

        self.nsteps = nsteps
        # x_gp : Input parameter for gaussian process
        self.x_gp = []
        # y_gp : Output target function
        self.y_gp = []

        self.target = target
        if kernel == 'matern':
            self.kernel = gp.kernels.Matern(nu=2.5)
        elif kernel == 'rbf':
            self.kernel = 1.0 * gp.kernels.RBF(length_scale=1.0)
        elif kernel == 'dotproduct':
            self.kernel = 1.0 * gp.kernels.DotProduct(sigma_0=1.0) ** 2

        self.model = gp.GaussianProcessRegressor(kernel=self.kernel, alpha=1e-5,
                                                 n_restarts_optimizer=25, normalize_y=True)

        # Constraints
        self.const = const
        if self.const:
            self.cons = [{'type': 'eq', 'fun': self.target.constraints}]

        self.next_value = 0
        self.best_ei = 0
        self.next_batch = []

        self.list_minyp = np.array([])
        self.list_minxp = np.array([])
        self.min_yp = None
        self.min_xp = None

        self.path_output = './output'

    def initialization(self):

        print('*** Initialization with %s random samples ***' % self.init_values)
        for i in range(self.dim):
            self.x_gp.append(np.random.uniform(self.bounds[i][0], self.bounds[i][1], size=self.init_values))

        self.x_gp = np.array(self.x_gp)
        self.x_gp = np.transpose(self.x_gp)
        print('Initial Input : \n', self.x_gp)

        pool = mp.Pool(processes=self.nb_procs)
        results = [pool.apply_async(self.target.target_f, args=(x, ib)) for (ib, x) in enumerate(self.x_gp)]
        results = [p.get() for p in results]
        results.sort()
        values = []
        for ib in range(self.x_gp.shape[0]):
            values.append(results[ib][1])

        self.y_gp = np.array(values)
        self.y_gp = self.y_gp.reshape(-1, 1)

        print('Initial Output : \n', self.y_gp)

        self.min_yp = np.min(self.y_gp)
        self.min_xp = self.x_gp[np.argmin(self.y_gp)]
        self.list_minyp = self.list_minyp.reshape(-1, 1)
        self.list_minxp = self.list_minxp.reshape(-1, self.dim)

        print('Initial best parameters: ', self.min_xp, 'minimum error: ', self.min_yp)

        print('*** End of Initialization ***')

    def bayesopt_step(self, batch_calc=False):

        reduction_done = 0
        same_min = 0

        if batch_calc and self.const:
            print('Batch calculations not available with constraints')
            batch_calc = False

        for step in range(self.nsteps):
            print("***************************")
            print("******** Step {0}/{1} ********".format(step + 1, self.nsteps))
            print("***************************")

            if batch_calc:
                self.maximize_batch_B3O()
            else:
                self.model.fit(self.x_gp, self.y_gp)
                self.search_next_value()

            for idx, val in enumerate(self.x_gp):
                if np.array_equal(val, self.next_value):  # check if a data point is already taken
                    print('next_value : ', self.next_value)
                    print("Random value for next sample")
                    self.next_value = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

            if batch_calc:
                print('Next size batch ', self.next_batch.shape[0])
                print('Next  batch ', self.next_batch)
            else:
                self.next_batch = np.array([self.next_value])
                print('Next size batch (1) ', self.next_batch.shape[0])
                print('Next  batch ', self.next_batch)

            print('next_value : ', self.next_value)

            pool = mp.Pool(processes=self.nb_procs)
            results = [pool.apply_async(self.target.target_f, args=(x, ib)) for (ib, x) in enumerate(self.next_batch)]
            results = [p.get() for p in results]
            results.sort()
            values = []
            for ib in range(self.next_batch.shape[0]):
                values.append(results[ib][1])
            results = np.array(values)

            if self.min_yp is None or np.min(results) < self.min_yp:
                same_min = 0
                self.min_yp = np.min(results).reshape(-1, 1)
                self.min_xp = self.next_batch[np.argmin(results)].reshape(-1, self.dim)
            else:
                same_min += 1

            self.x_gp = np.append(self.x_gp, self.next_batch.reshape(-1, self.dim), axis=0)
            self.y_gp = np.append(self.y_gp, results.reshape(-1, 1), axis=0)
            self.list_minyp = np.append(self.list_minyp, self.min_yp.reshape(-1, 1), axis=0)
            self.list_minxp = np.append(self.list_minxp, self.next_value.reshape(-1, self.dim), axis=0)

            # ######### ########
            # Plotting iteration
            # ######### ########

            if self.dim == 2:
                os.makedirs(self.path_output, exist_ok=True)
                self.plot_iteration2d(step)
            else:
                print('No plotting available')

            # ######### ######## ######## ########
            # Search space reduction after 10 steps
            # Done 2 times maximum
            # ######### ######## ######## ########

            if same_min == 10 and reduction_done < 2:
                print('Same minimum during 10 steps !')
                print('Search space reduction !')
                print("For x = ", np.squeeze(self.min_xp), ", Current Minimum = ", np.squeeze(self.min_yp))

                new_bounds = []
                self.min_xp = np.reshape(self.min_xp, (1, self.dim))

                for d in range(self.dim):
                    limit = (self.bounds[d][1] - self.bounds[d][0]) / 7
                    new_bounds.append([max(self.bounds[d][0], self.min_xp[0][d] - limit),
                                       min(self.bounds[d][1], self.min_xp[0][d] + limit)])

                self.bounds = new_bounds
                self.bounds = np.array(self.bounds)

                print('new bounds: ', self.bounds)
                reduction_done += 1

            # ######### ########
            # Convergence check
            # ######### ########

            # if self.min_yp < self.epsilon:
            #    print('Criteria convergence reached ! End of optimization at step {0}'.format(step))
            #    print('error: ', self.min_yp)

            # ######### ########
            # End of optimization
            # ######### ########

        # if self.min_yp > self.epsilon:
        #    print('Criteria convergence not reached !')
        print('Maximum number of steps {0} reached.'.format(self.nsteps))

        print('*** End of Bayesian Optimization ***')
