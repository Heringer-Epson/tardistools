# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
#  File Name : tardis_minimal_model.py
#
#  Purpose :
#
#  Creation Date : 18-02-2016
#
#  Last Modified : Tue 02 Aug 2016 15:09:26 CEST
#
#  Created By :
#
# _._._._._._._._._._._._._._._._._._._._._.
"""This module provides an interface object which holds the essential
information of a Tardis run to do all the diagnostic tasks for which the
tardisanalysis repository provides tools. Relying on this interface model
objects is a temporary solution until the model storage capability of Tardis
has reached a mature state.
"""
import pandas as pd
import astropy.units as units
import os
import logging
from tardis.model import Radial1DModel

logger = logging.getLogger(__name__)

class minimal_model(object):
    """Interface object used in many tardisanalysis tools. It holds the
    essential diagnostics information for either the real or the virtual packet
    population of a run.

    This interface object may be filled from an existing Tardis radial1dmodel
    object (for example during the interactive ipython use of Tardis), or
    filled from an HDF5 file (generated by store_data_for_minimal_model).

    Parameters
    ----------
    mode : str
        "real" (default) or "virtual"; defines which packet population is
        stored in the interface object.
    """
    def __init__(self, mode="real"):

        allowed_modes = ["real", "virtual"]
        try:
            assert(mode in allowed_modes)
        except AssertionError:
            msg = "unknown mode '{:s}';".format(mode) + \
                "allowed modes are {:s}".format(",".join(allowed_modes))
            raise ValueError(msg)

        self.readin = False
        self.mode = mode
        self.lines = None
        self.last_interaction_type = None
        self.last_line_interaction_in_id = None
        self.last_line_interaction_out_id = None
        self.last_interaction_in_nu = None
        self.packet_nus = None
        self.packet_energies = None
        self.spectrum_wave = None
        self.spectrum_luminosity = None
        self.time_of_simulation = None

    def from_interactive(self, simulation):
        """fill the minimal_model from an existing radial1dmodel object

        Parameters
        ----------
        mdl : Radial1DModel
            Tardis model object holding the run
        """
        
        self.time_of_simulation = simulation.runner.time_of_simulation
        self.lines = simulation.atom_data.lines

        if self.mode == "virtual":

            self.last_interaction_type = \
                simulation.runner.virt_packet_last_interaction_type
            self.last_line_interaction_in_id = \
                simulation.runner.virt_packet_last_line_interaction_in_id
            self.last_line_interaction_out_id = \
                simulation.runner.virt_packet_last_line_interaction_out_id
            self.last_interaction_in_nu = \
                simulation.runner.virt_packet_last_interaction_in_nu
            self.packet_nus = \
                simulation.runner.virt_packet_nus * units.Hz
            self.packet_energies = \
                simulation.runner.virt_packet_energies * units.erg
            self.spectrum_wave = \
                simulation.runner.spectrum_virtual.wavelength
            self.spectrum_luminosity = \
                simulation.runner.spectrum_virtual.luminosity_density_lambda
        elif self.mode == "real":

            esc_mask = simulation.runner.output_energy >= 0

            self.last_interaction_type = \
                simulation.runner.last_interaction_type[esc_mask]
            self.last_line_interaction_in_id = \
                simulation.runner.last_line_interaction_in_id[esc_mask]
            self.last_line_interaction_out_id = \
                simulation.runner.last_line_interaction_out_id[esc_mask]
            self.last_interaction_in_nu = \
                simulation.runner.last_interaction_in_nu[esc_mask]
            self.packet_nus = \
                simulation.runner.output_nu[esc_mask]
            self.packet_energies = \
                simulation.runner.output_energy[esc_mask]
            self.spectrum_wave = \
                simulation.runner.spectrum.wavelength
            self.spectrum_luminosity = \
                simulation.runner.spectrum.luminosity_density_lambda
        else:
            raise ValueError
        self.last_interaction_in_nu = self.last_interaction_in_nu * units.Hz
        self.readin = True