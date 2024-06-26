# Radiation Synthesis Tools

These are radiation synthesis tools written in Python and accelerated using cython.
They could be used to analyze data simulated by [MPI-AMRVAC](https://github.com/amrvac/amrvac), 
but also applicable to other relevant data.
> An early version is also available in [radsyn_tools][https://github.com/gychen-NJU/radsyn_tools]. We fix some problems in the early version and accomplish the real spherical synthesis.

They have the following features:
- Synthesizing optically thin radiation in the Extreme Ultra-Violet (EUV) wavebands, including SDO/AIA 171, 304, 193, 335, 211, 94, 131 Angstrom.
- Synthesizing optically thick radiation in the Extreme Ultra-Violet (EUV) wavebands, including SDO/AIA 171, 304, 193, 335, 211, 94, 131 Angstrom.
- Synthesizing optically thick radiation in the Radio waveband.
- Easily to extend physical modules.
- Interactive GUI operations of synthesized images.
- Multi perspective views with camera controls.
