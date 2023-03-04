# GMI-MST

This is a multi-channel image registration framework by using multifeature mutual information based on (weighted) minimal spanning tree. The two methods (called LaMI and LWaMI) can be faster using ITK multi-threads.

The source code includes two folders: Metrics and ImageSamplers. The users can embed them into elastix platform by using cmake. To be convenient, an executable file based on Windows system can be found from folder 'Release'. 

An example of intrasubject registration on cardiac MR is illustrated in folder 'examples'. The "patient001_frame01.nii.gz" is the fixed image. The "patient001_frame12.nii.gz" is the moving image. 
The deep features of fixed and moving images can be found from folder 'examples/dlfs'. The spin features of fixed image can be found from folder 'examples/spin'. The Moran's I coefficient image can be found from folder 'examples/moran'. 

Also, in folder 'examples' the users can find the parameter file "parameters.LMST.ft.bs.txt" for the LaMI method, and "parameters.LMST.pm.ft.bs.txt" for the LWaMI method.


# usage

The command line to run a registration for LaMI is as follows:
elastix -f0 fixed0.ext -f1 fixed1.ext ... -f16 fixed16.ext -m0 moving0.ext -m1 moving1.ext ... -m16 moving16.ext -out outDir -p parameters.LMST.ft.bs.txt

The command line to run a registration for LWaMI is as follows:
elastix -f0 fixed0.ext -f1 fixed1.ext ... -f16 fixed16.ext -m0 moving0.ext -m1 moving1.ext ... -m16 moving16.ext -pw32 spin1.ext -pw33 spin2.ext ... -pw47 spin16.ext -out outDir -p parameters.LMST.pm.ft.bs.txt -fmoran2 moran.ext

The users can find the log files "elastix-lami.log" for LaMI, and "elastix-lwami.log" for LWaMI running on a 8-core computer.
