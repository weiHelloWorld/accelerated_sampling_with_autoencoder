#!/bin/bash

# this script sync between server and local machine

echo "sync files between local machine and ALF server"
echo "-----------------------------------------------"

echo "-----------------------------------------------"
echo "sending src"
echo "-----------------------------------------------"
echo ""
rsync -urv ~/Dropbox/MD_simulation_research/MD_simulation_on_alanine_dipeptide/current_work/src weichen9@alf-clustersrv.mrl.illinois.edu:~/current_work    # transfer src folder to ALF

echo "-----------------------------------------------"
echo "receiving resources"
echo "-----------------------------------------------"
echo ""
rsync -urv weichen9@alf-clustersrv.mrl.illinois.edu:~/current_work/resources ~/Dropbox/MD_simulation_research/MD_simulation_on_alanine_dipeptide/current_work/    # get back resources ALF

echo "-----------------------------------------------"
echo "receiving target"
echo "-----------------------------------------------"
echo ""
rsync -urv weichen9@alf-clustersrv.mrl.illinois.edu:~/current_work/target ~/Dropbox/MD_simulation_research/MD_simulation_on_alanine_dipeptide/current_work/    # get back target ALF
