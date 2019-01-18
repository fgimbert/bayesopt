import numpy as np
from scipy.optimize import minimize
from sklearn import mixture
from sklearn.metrics.pairwise import euclidean_distances


class Batch:

    def test_batch(self):
        print(self.x_gp[0])

    def acq_batch_generalized_slice_sampling_generate(self, npoints, SliceBlock=200):
        """
        A Batch Generalized Slice Sampling technique to draw self.y_gp under the acquisition function
        Input Parameters
        ----------
        acq_fun: The acquisition function object that return its point-wise value.
        gp: A gaussian process fitted to the relevant data.
        target_min: The current minimum known value of the target function. (y_max)
        bounds: The  bounds to limit the search of the acq max.
        N: number of points to generate
        SliceBlock: block size for batch slice sampling

        Returns
        -------
        self.y_gp
        """

        npoints = npoints * self.dim

        # find minimum value in acquisition function
        # Calculate EI on grid value first

        dimension = self.dim
        if dimension > 4:
            grid = 10
        else:
            grid = 20

        dimensions = [np.linspace(self.bounds[i][0], self.bounds[i][1], grid) for i in range(dimension)]
        param_grid = np.array(np.meshgrid(*dimensions)).T.reshape(-1, dimension)
        ei = []
        for x in param_grid:
            # acq_fun returns -1 * acq_fun -> need to multiply by -1 to find the minimum
            current_ei = -1 * self.acq_function(x)
            ei.append(current_ei)

        ei = np.array(ei)
        ei = ei.reshape((-1, 1))
        ei = ei.reshape([grid] * dimension)
        # print(ei.shape)
        min_acq = np.min(ei)
        x_min = param_grid[np.argmin(ei)]

        # EI is zero at most values -> often get trapped
        # in a local minimum -> multistarting to increase
        # our chances to find the global minimum

        starting_points = []
        for i in range(self.dim):
            starting_points.append(np.random.uniform(self.bounds[i][0], self.bounds[i][1], size=200))

        starting_points = np.array(starting_points)
        starting_points = np.transpose(starting_points)
        np.append(starting_points, x_min)

        for point in starting_points:
            point = point.reshape(1, -1)
            # acq_fun returns -1 * acq_fun -> need to multiply by -1 to find the minimum
            sign = -1
            res = minimize(fun=self.acq_function, x0=point,
                           bounds=self.bounds,
                           method='L-BFGS-B', args=[sign])

            if res.fun < min_acq:
                min_acq = res.fun
                x_min = res.x

        #     counter for #self.y_gp rejected
        count_reject = 0
        count = 0
        batch = []

        # init x0
        x0 = np.zeros(self.dim)
        for idx in range(0, self.dim):
            x0[idx] = np.random.uniform(self.bounds[idx][0], self.bounds[idx][1], 1)

        # evaluate alpha(x0) from the acquisition function
        fx = -1 * self.acq_function(x0)
        fx = fx.ravel()
        fx = np.dot(fx, np.ones((1, SliceBlock)))
        idxAccept = range(0, SliceBlock)
        y = fx
        cut_min = np.dot(min_acq, np.ones((1, SliceBlock)))
        cut_min = cut_min.squeeze()

        while count < npoints:
            # sampling y

            # make a threshold (cut_min) to draw self.y_gp under the peaks, but above the threshold
            for idx in range(0, SliceBlock):
                if idx in idxAccept:
                    temp = np.linspace(min_acq, fx[idx], 100)
                    temp = temp.ravel()
                    cut_min[idx] = np.percentile(temp, 85)

            y[idxAccept] = np.random.uniform(cut_min[idxAccept], fx[idxAccept], len(idxAccept))

            # sampling x
            x = np.zeros((SliceBlock, self.dim))
            for idx in range(0, self.dim):
                x[:, idx] = np.random.uniform(self.bounds[idx][0], self.bounds[idx][1], SliceBlock)

            # get f(x)=alpha(x)
            fx = []
            for point in x:
                fx.append(-1 * self.acq_function(point))
            fx = np.array(fx)
            fx = fx.ravel()

            idxAccept = [idx for idx, val in enumerate(fx) if val > cut_min[idx] and val > y[idx]]

            if len(batch) == 0:
                batch = x[idxAccept, :]
            else:
                batch = np.vstack((batch, x[idxAccept, :]))
            count = len(batch)
            count_reject = count_reject + len(fx) - len(idxAccept)

            # stop the sampling process if #rejected self.y_gp excesses a threshold
            if count_reject > npoints * 10:
                # print 'BGSS count_reject={:d}, count_accept={:d}'.format(count_reject,count)

                if count < 5:
                    batch = []
                return batch

        return np.asarray(batch)

    def fitIGMM(self, obs):
        """
        Fitting the Infinite Gaussian Mixture Model and GMM where applicable
        Input Parameters
        ----------

        obs:        self.y_gp  generated under the acqusition function by BGSS

        IsPlot:     flag variable for visualization


        Returns
        -------
        mean vector: mu_1,...mu_K
        """

        if self.dim <= 2:
            n_init_components = 3
        else:
            n_init_components = np.int(self.dim * 1.1)

        dpgmm = mixture.BayesianGaussianMixture(weight_concentration_prior_type='dirichlet_process',
                                                n_components=n_init_components, covariance_type="full")
        dpgmm.fit(obs)

        # check if DPGMM fail, then use GMM.
        mydist = euclidean_distances(dpgmm.means_, dpgmm.means_)
        np.fill_diagonal(mydist, 99)

        if dpgmm.converged_ is False or np.min(mydist) < (0.01 * self.dim):
            dpgmm = mixture.GaussianMixture(n_components=n_init_components, covariance_type="full")
            dpgmm.fit(obs)

            # truncated for variational inference
        weight = dpgmm.weights_
        weight_sorted = np.sort(weight)
        weight_sorted = weight_sorted[::-1]
        temp_cumsum = np.cumsum(weight_sorted)

        cutpoint = 0
        for idx, val in enumerate(temp_cumsum):
            if val > 0.7:
                cutpoint = weight_sorted[idx]
                break

        ClusterIndex = [idx for idx, val in enumerate(dpgmm.weights_) if val >= cutpoint]

        myMeans = dpgmm.means_[ClusterIndex]
        # dpgmm.means_=dpgmm.means_[ClusterIndex]
        dpgmm.truncated_means_ = dpgmm.means_[ClusterIndex]

        new_X = myMeans.reshape((len(ClusterIndex), -1))
        new_X = new_X.tolist()

        return new_X

    def maximize_batch_B3O(self):
        """
        Finding a batch of points using Budgeted Batch Bayesian Optimization approach

        Input Parameters
        ----------
        gp_params:          Parameters to be passed to the Gaussian Process class

        kappa:              constant value in UCB

        IsPlot:             flag variable for visualization

        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """

        # Step 2 in the Algorithm

        # Set parameters for Gaussian Process

        # print("Prior (kernel:  %s)" % kernel)
        self.model.fit(self.x_gp, self.y_gp)
        # print("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
        #   % (model.kernel_, model.log_marginal_likelihood(model.kernel_.theta)))
        # print('End fit')
        # Search next value
        self.search_next_value()

        # Step 4 in the Algorithm
        # generate self.y_gp from Acquisition function

        # check the bound 0-1 or original bound
        obs = self.acq_batch_generalized_slice_sampling_generate(npoints=500)

        # Step 5 and 6 in the Algorithm
        batch = []
        batch_igmm = []

        if len(obs) == 0:  # monotonous acquisition function
            print("Monotonous acquisition function")
            for i in range(self.dim):
                batch.append(np.random.uniform(self.bounds[i][0], self.bounds[i][1], size=self.batch_size - 1))

            batch = np.array(batch)
            batch = np.transpose(batch)
            # batch = np.append(batch, self.next_value.reshape(1, self.dim), axis=0)

        else:
            batch_igmm = self.fitIGMM(obs)

        # Test if x_max is repeated, if it is, draw another one at random
        for idx, val in enumerate(batch_igmm):
            if np.all(np.any(np.abs(self.x_gp - val) > 0.02, axis=1)):  # check if a data point is already taken
                batch = np.append(batch, val)

        if len(batch) == 0:
            batch = np.zeros((1, self.dim))
            for idx in range(0, self.dim):
                batch[0, idx] = np.random.uniform(self.bounds[idx, 0], self.bounds[idx, 1], 1)
        else:
            batch = batch.reshape((-1, self.dim))

        # Limit batch to batch_size -1 + next_value
        if batch.shape[0] < self.batch_size - 1:
            print('batch shape: ', batch.shape[0])
            batch = np.append(batch, self.next_value.reshape(1, self.dim), axis=0)
        else:
            batch = batch[:self.batch_size - 2]
            print('batch shape: ', batch.shape[0])
            batch = np.append(batch, self.next_value.reshape(1, self.dim), axis=0)

        self.next_batch = batch
