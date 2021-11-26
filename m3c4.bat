@Echo Off
call conda activate tensorflow

call set PYTHONHOME=C:\tools\miniconda3\envs\tensorflow
call cd m3c4
call cargo run --example %1 --release

pause