#! /bin/bash
jobid=35
cp -r experiment experiment$jobid
cd experiment$jobid
mv script script$jobid
qsub -v jobid=$jobid script$jobid
