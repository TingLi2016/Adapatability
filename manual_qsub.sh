#!/bin/bash

echo "[start]"
echo `date`



### for test set 1997-1998
qsub -q hpc.q@ncshpcgpu01 methods.sh 1999_2001_1997_1998 1997 1998 1999 2001 1999 2001
qsub -q hpc.q@ncshpcgpu01 methods.sh 1999_2004_1997_1998 1997 1998 1999 2004 1999 2004
qsub -q hpc.q@ncshpcgpu01 methods.sh 1999_2007_1997_1998 1997 1998 1999 2007 1999 2007
qsub -q hpc.q@ncshpcgpu01 methods.sh 1999_2019_1997_1998 1997 1998 1999 2019 1999 2019


### for test set 1999-2001
qsub -q hpc.q@ncshpc410 methods.sh 1997_1998_1999_2001 1999 2001 1997 1998 1997 1998
qsub -q hpc.q@ncshpc410 methods.sh 1997_2004_1999_2001 1999 2001 1997 1998 2002 2004
qsub -q hpc.q@ncshpc410 methods.sh 1997_2007_1999_2001 1999 2001 1997 1998 2002 2007
qsub -q hpc.q@ncshpc410 methods.sh 1997_2019_1999_2001 1999 2001 1997 1998 2002 2019


### for test set 2002-2004
qsub -q hpc.q@ncshpc400 methods.sh 1997_1998_2002_2004 2002 2004 1997 1998 1997 1998
qsub -q hpc.q@ncshpc400 methods.sh 1997_2001_2002_2004 2002 2004 1997 2001 1997 2001
qsub -q hpc.q@ncshpc400 methods.sh 1997_2007_2002_2004 2002 2004 1997 2001 2005 2007
qsub -q hpc.q@ncshpc400 methods.sh 1997_2019_2002_2004 2002 2004 1997 2001 2005 2019


### for test set 2005-2007
qsub -q hpc.q@ncshpc401 methods.sh 1997_2019_2005_2007 2005 2007 1997 2004 2008 2019
qsub -q hpc.q@ncshpc401 methods.sh 1997_2004_2005_2007 2005 2007 1997 2004 1997 2004
qsub -q hpc.q@ncshpc401 methods.sh 1997_2001_2005_2007 2005 2007 1997 2001 1997 2001
qsub -q hpc.q@ncshpc401 methods.sh 1997_1998_2005_2007 2005 2007 1997 1998 1997 1998


### for test set 2008-2019
qsub -q hpc.q@ncshpc407 methods.sh 1997_1998_2008_2019 2008 2019 1997 1998 1997 1998
qsub -q hpc.q@ncshpc407 methods.sh 1997_2001_2008_2019 2008 2019 1997 2001 1997 2001
qsub -q hpc.q@ncshpc407 methods.sh 1997_2004_2008_2019 2008 2019 1997 2004 1997 2004
qsub -q hpc.q@ncshpc407 methods.sh 1997_2007_2008_2019 2008 2019 1997 2007 1997 2007




exit


### example
#for i in {0..5}
#for i in mold2_1 mold2_2 mold2_3 mold2_4
#for i in $(seq 50 50 100)
do
  qsub -q hpc.q@ncshpc4* methods.sh $i
done




