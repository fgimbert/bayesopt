import argparse
from BayesOpt import *
from TargetFunction import *


# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
# Arguments
parser.add_argument('--dim', type=int, default=2, help='dimension')
parser.add_argument('--nsteps', type=int, default=10, help='dimension')

parser.add_argument('--test', type=bool, default=True, help='test')
parser.add_argument('--const', type=bool, default=True, help='constraints')
parser.add_argument('--batch', type=bool, default=False, help='batch calculation')
args = parser.parse_args()


def main():
    print('***Welcome in the Bayesian Optimization Code !***\n')
    print("Optimization parameters:")
    print('Dimension: ', args.dim)
    print('Maximum steps: ', args.nsteps)
    print('Test : ', args.test)

    target_function = TargetFunction(dimension=args.dim, test=args.test)
    bayesopt = BayesOpt(nsteps=args.nsteps, target=target_function, const=args.const, bounds=[[0, 1], [0, 1]])

    bayesopt.initialization()
    bayesopt.bayesopt_step(batch_calc=args.batch)

    bayesopt.plot_convergence()
    print('Minimum : ', np.squeeze(bayesopt.min_yp))
    print('Parameters : ', bayesopt.min_xp)


if __name__ == '__main__':
    main()
