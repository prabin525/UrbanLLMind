"""
Entrypoint for the Urban Agent Simulation.
Parses command line arguments, initializes the model parameters,
and starts the Repast4Py simulation runner.
"""
from mpi4py import MPI
from repast4py import parameters
from mini_world.model import Model


def run(params):
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
