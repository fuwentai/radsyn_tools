# Radiation Synthesis Tools

These are radiation synthesis tools written in Python and accelerated using cython.
They could be used to analyze data simulated by [MPI-AMRVAC](https://github.com/amrvac/amrvac), 
but also applicable to other relevant data.
> An early version is also available in [radsyn_tools](https://github.com/gychen-NJU/radsyn_tools), and you can use the codes to transfer the data from sphere to cartesian to use the method 1 in this version. In this version, we solve the radiation transfer equation and accomplish the real spherical synthesis using method 2..

They have the following features:
- Synthesizing optically thick radiation in the Extreme Ultra-Violet (EUV) wavebands, including SDO/AIA 171, 304, 193, 335, 211, 94, 131 Angstrom. Optically thin radiation is included in thick radiation when absorption chi equals zero.
- Synthesizing optically thick radiation in the Radio waveband.
- Synthesizing optically thin white light corona.
- Easily to extend physical modules.
- Interactive GUI operations of synthesized images.
- Multi perspective views with camera controls.


Three example Figures are uploaded.
