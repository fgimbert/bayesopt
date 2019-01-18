import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator


class Plot:

    def plot_iteration2d(self, step):
        grid = 75
        dim1 = np.linspace(self.bounds[0][0], self.bounds[0][1], grid, endpoint=True)
        dim2 = np.linspace(self.bounds[1][0], self.bounds[1][1], grid, endpoint=True)
        # We need the cartesian combination of these two vectors
        param_grid = np.array([[x1, x2] for x2 in dim2 for x1 in dim1])
        eps, sig = np.meshgrid(dim1, dim2)
        mean, std = self.model.predict(param_grid, return_std=True)

        ei = []
        for x in param_grid:
            current_ei = -1 * self.acq_function(x)
            ei.append(current_ei)

        ei = np.array(ei)
        ei = ei.reshape((-1, 1))
        ei = ei.reshape(eps.shape)
        print('next value plot', param_grid[np.argmax(ei)])
        fig, (ax2, ax1) = plt.subplots(2, 1)
        fig.suptitle('Gaussian Process and Improvement Function Step %d' % (step + 1))
        # EI contour plot
        # cp = ax1.contourf(eps, sig, ei, 20, cmap='jet')
        cp = ax1.imshow(ei, cmap='jet', extent=[np.min(eps), np.max(eps), np.min(sig), np.max(sig)],
                        interpolation='bilinear', origin='lower')

        plt.colorbar(cp, ax=ax1)
        # ax1.set_title("Expected Improvement. Next sample will be (%.4f, %.4f)" % (best_x[0], best_x[1]))
        ax1.autoscale(False)
        # ax1.axvline(best_x[0], color='k')
        # ax1.axhline(best_x[1], color='k')
        for batch_i in range(self.next_batch.shape[0]):
            ax2.scatter(self.next_batch[batch_i][0], self.next_batch[batch_i][1], marker='*', color='black')
        ax1.set_xlabel("Parameter 1")
        ax1.set_ylabel("Parameter 2")
        ax1.set_xlim(self.bounds[0][0], self.bounds[0][1])
        ax1.set_ylim(self.bounds[1][0], self.bounds[1][1])
        ax1.set_xbound(self.bounds[0][0], self.bounds[0][1])
        ax1.set_ybound(self.bounds[1][0], self.bounds[1][1])
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1, nbins=4))
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1, nbins=4))
        ax1.set_aspect('equal')

        # Loss contour plot
        # cp2 = ax2.contourf(eps, sig, mean.reshape(eps.shape), 20, cmap='RdBu')
        cp2 = ax2.imshow(mean.reshape(eps.shape), cmap='RdBu',
                         extent=[np.min(eps), np.max(eps), np.min(sig), np.max(sig)],
                         interpolation='bilinear', origin='lower')

        plt.colorbar(cp2, ax=ax2)
        ax2.autoscale(False)
        ax2.scatter(self.x_gp[:self.init_values, 0], self.x_gp[:self.init_values, 1], zorder=1, marker='h', color='blue', s=14)
        ax2.scatter(self.x_gp[self.init_values:, 0], self.x_gp[self.init_values:, 1], zorder=1, marker='h', color='red', s=14)
        # ax2.axvline(best_x[0], color='k')
        # ax2.axhline(best_x[1], color='k')
        for batch_i in range(self.next_batch.shape[0]):
            ax2.scatter(self.next_batch[batch_i][0], self.next_batch[batch_i][1], marker='*', color='black')
        # ax2.set_title("Mean estimate of loss surface for iteration %d" % (sampled_params.shape[0]))
        # ax2.set_xlabel("Parameter 1")
        ax2.set_xlim(self.bounds[0][0], self.bounds[0][1])
        ax2.set_ylim(self.bounds[1][0], self.bounds[1][1])
        ax2.set_xbound(self.bounds[0][0], self.bounds[0][1])
        ax2.set_ybound(self.bounds[1][0], self.bounds[1][1])
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1, nbins=4))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1, nbins=4))

        ax2.set_ylabel("Parameter 2")
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        # ax2.scatter(0.4615, 0.4155, marker='*', c='gold')
        ax2.set_aspect('equal')
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

        plt.savefig("%s/iteration_%d.png" % (self.path_output, step+1))
        plt.savefig("%s/iteration_%d.eps" % (self.path_output, step+1))

        plt.cla()
        plt.close(fig)

    def plot_convergence(self):

        output = '%s/bayesopt.out' % self.path_output
        print(self.x_gp.shape, self.x_gp.shape)
        np.savetxt(output, np.c_[self.x_gp, self.x_gp], fmt='%f', delimiter='\t')

        distance = []
        steps = []
        for i in range(len(self.list_minxp[:-1])):
            steps.append(i)
            distance.append(np.linalg.norm(self.list_minxp[i] - self.list_minxp[i + 1]))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        np.set_printoptions(precision=4, suppress=True)
        fig.suptitle('Best Minimum found at x =  %s ' % np.squeeze(self.min_xp))

        ax1.plot(distance, 'ro-', markersize=4)
        ax1.autoscale(False)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Distance between consecutive parameters")
        ax1.set_xlim(0, self.nsteps)
        ax1.set_xbound(0, self.nsteps)

        ax2.plot(self.list_minyp, 'bo-', markersize=4)
        ax2.autoscale(False)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Minimum error")
        ax2.set_xlim(0, self.nsteps)
        ax2.set_xbound(0, self.nsteps)

        plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
        plt.savefig("%s/convergence.png" % self.path_output)
        plt.savefig("%s/convergence.eps" % self.path_output)

        plt.cla()
        plt.close(fig)

        return fig

