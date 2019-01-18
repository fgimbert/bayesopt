from scipy import stats
import numpy as np
from scipy.optimize import minimize


class BayesOptTools:

    def expected_improvement(self, x, sign=1):
        # are we trying to maximise a score or minimise an error?
        x_to_predict = x.reshape(-1, self.dim)
        # print('Samples shape : ', samples.shape)
        # print('x_to_predict shape : ', x_to_predict.shape)

        if self.find_max:
            best_sample = self.y_gp[np.argmax(self.y_gp)]
            mean, cov = self.model.predict(x_to_predict, return_cov=True)
            sigma = np.sqrt(cov.diagonal())
            z = (mean - best_sample - self.xi) / sigma
            ei = ((mean - best_sample - self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z))
            # minimize -ei to find the maximum of ei.

            return -ei
        else:
            best_y = self.y_gp[np.argmin(self.y_gp)]
            mean, cov = self.model.predict(x_to_predict, return_cov=True)
            sigma = np.sqrt(cov.diagonal())
            z = (best_y - mean - self.xi) / sigma
            ei = ((best_y - mean - self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z))

            # minimize -ei to find the maximum of ei.
            return sign * (-ei)

    def search_next_value(self):
        """Find point with largest expected improvement"""

        # Calculate EI on grid value first
        if self.dim > 4:
            grid = 10
        else:
            grid = 20

        dimensions = [np.linspace(self.bounds[i][0], self.bounds[i][1], grid) for i in range(self.dim)]
        param_grid = np.array(np.meshgrid(*dimensions)).T.reshape(-1, self.dim)
        ei = []
        for x in param_grid:
            current_ei = -1 * self.acq_function(x)
            ei.append(current_ei)

        ei = np.array(ei)
        ei = ei.reshape((-1, 1))
        ei = ei.reshape([grid] * self.dim)
        best_improvement_value = -1 * np.max(ei)
        best_x = param_grid[np.argmax(ei)]
        # print('grid next value: ', best_x, 'ei: ', best_improvement_value)

        # EI is zero at most values -> often get trapped
        # in a local maximum -> multistarting to increase
        # our chances to find the global maximum

        starting_points = []
        for i in range(self.dim):
            starting_points.append(np.random.uniform(self.bounds[i][0], self.bounds[i][1], size=200))

        starting_points = np.array(starting_points)
        starting_points = np.transpose(starting_points)

        if self.const:
            res = minimize(fun=self.acq_function, x0=best_x,
                           bounds=self.bounds,
                           method='SLSQP',
                           constraints=self.cons)

            best_improvement_value = res.fun
            best_x = res.x

        for point in starting_points:
            point = point.reshape(1, -1)
            if self.const:
                res = minimize(fun=self.acq_function, x0=point,
                               bounds=self.bounds,
                               method='SLSQP',
                               constraints=self.cons)

            else:
                res = minimize(fun=self.acq_function, x0=point,
                               bounds=self.bounds,
                               method='L-BFGS-B')

            if res.fun < best_improvement_value:
                best_improvement_value = res.fun
                best_x = res.x

        self.next_value = best_x
        self.best_ei = best_improvement_value
