import numpy as np
import os

os.environ['DEVITO_OPENMP'] = '1'

from argparse import ArgumentParser

from devito.logger import info
from devito import Constant, Function, smooth
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry, Model
import os
import sys

def parse_args(argv):
    """Parse Commandline Arguments
    
    Arguments:
        argv {dict} -- Contains command line arguments
    
    Returns:
        ArgparseArguments -- Parsed commandline arguments
    """

    description = ("Wave-solver for Roeth and Tarantola 1994 Example")
    parser = ArgumentParser(description=description)
    parser.add_argument('-nd', dest='ndim', default=2, type=int,
                        help="Preset to determine the number of dimensions")
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=20,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("--input", type=str, help="Name of input velocity model file.")
    parser.add_argument("--output", type=str, help="Name of output waveform model file.")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")

    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DLE) mode")
    parser.add_argument("--checkpointing", default=False, action='store_true',
                        help="Use Checkpointing?")
    args = parser.parse_args(argv)

    return args 

def acoustic_setup(vp, shape=(500, 360), spacing=(5.0, 5.0),
                    tn=2710., kernel='OT2', space_order=6, nbpml=20,
                    dist_first=140, rec_spacing=90, 
                    num_rec=20, peak_freq=8.,
                **kwargs):
    """Setup the basic geometry for the wave forward propagator
    
    Arguments:
        vp {np.array} -- 2D-array of p-Wave Velocity
    
    Keyword Arguments:
        shape {tuple} -- Extents of the domain in number of gridblocks (default: {(500, 360)})
        spacing {tuple} -- Dimensions of gridblocks in x and y in [m] (default: {(5.0, 5.0)})
        tn {[type]} -- Total simulation runtime (default: {2710.})
        kernel {str} -- Discretization kernel in time (default: {'OT2'})
        space_order {int} -- Discretization kernel in space (default: {6})
        nbpml {int} -- Number of boundary layer gridblocks (default: {20})
        dist_first {int} -- Distance to first receiver (default: {140})
        rec_spacing {int} -- Spacing between all receivers (default: {90})
        num_rec {int} -- Number of receivers (default: {20})
        peak_freq {[type]} -- Peak frequency of Ricker wavelet in [Hz] (default: {8.})
    
    Returns:
        AcousticWaveSover -- Initialised solver object
    """

    #Define the basic Model container
    origin = kwargs.pop('origin', tuple([0. for _ in shape]))
    dtype = kwargs.pop('dtype', np.float32)

    model = Model(space_order=space_order, vp=vp.T, origin=origin, shape=shape, dtype=kwargs.pop('dtype', np.float32), spacing=spacing, nbpml=nbpml, **kwargs)
    
    # Source and receiver geometries
    src_coordinates = np.zeros((1, len(spacing)))    
    rec_coordinates = np.zeros((num_receivers, len(spacing)))
    rec_coordinates[:, 0] = dist_first+rec_spacing*np.array(range(num_rec))

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0=0.0, tn=tn, src_type='Ricker', f0=peak_freq/1000.)
    
    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, geometry, kernel=kernel, space_order=space_order, **kwargs)

    return solver


def run(vp, full_run=False, autotune=False, checkpointing=False, **kwargs): 
    """Setup and run a forward solver for the acoustic wave problem
    
    Arguments:
        vp {np.array} -- 2D-array of p-Wave Velocity
    
    Keyword Arguments:
        full_run {bool} -- Whether to run and store full waveforms (default: {False})
        autotune {bool} -- Use devito autotuning? (default: {False})
        checkpointing {bool} -- Use devito checkpointing? (default: {False})
    
    Returns:
        rec -- Receivers and recorded wavefield at receivers
    """

    # Define receiver geometry (spread across x, just below surface)
    solver = acoustic_setup(vp, **kwargs)

    info("Applying Forward")
    save = full_run and not checkpointing

    rec, u, summary = solver.forward(save=save, autotune=autotune)

    return rec

if __name__ == "__main__":
    #parse parameters
    args = parse_args(sys.argv[1:])

    #Simulation Parameters
    tn = 2710.0
    num_samples = 271

    #Geometry Parameters
    shape = tuple([500, 360])
    spacing = tuple([5.0, 5.0])
    dist_first_receiver = 140. #meters
    spacing_receivers = 90. #meters
    num_receivers = 20

    #Wavelet Parameters
    peak_frequency = 8. #Hz

    vp = np.load(os.path.expandvars(args.input)).astype(np.float32)/1000. #convert from m/s to km/s
    amps = []
    for i, v in enumerate(vp):
        
        print("Computing Model ", i)
        rec = run(v, shape=shape, spacing=spacing, nbpml=args.nbpml, tn=tn,
                space_order=args.space_order, kernel=args.kernel,
                autotune=args.autotune, dse=args.dse, dle=args.dle, full_run=args.full,
                checkpointing=args.checkpointing, 
                dist_first=dist_first_receiver, rec_spacing=spacing_receivers, 
                num_rec=num_receivers, peak_freq=peak_frequency)
        
        #Resample to 'num_samples' samples and store
        amps.append(rec.resample(num=num_samples).data)
    
    #Save to disk
    np.save(os.path.expandvars(args.output), np.array(amps))
