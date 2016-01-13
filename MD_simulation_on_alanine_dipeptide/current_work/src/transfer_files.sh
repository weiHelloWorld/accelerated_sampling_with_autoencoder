#!/bin/bash

# this script sync between server and local machine

PROJECT_DIR="$HOME/Dropbox/temp_Linux/MD_simulation_research"

echo "sync files between local machine and ALF server"
echo "-----------------------------------------------"

echo "-----------------------------------------------"
echo "sending src"
echo "-----------------------------------------------"
echo ""
rsync -urv ${PROJECT_DIR}/MD_simulation_on_alanine_dipeptide/current_work/src weichen9@alf-clustersrv.mrl.illinois.edu:~/current_work    # transfer src folder to ALF

echo "-----------------------------------------------"
echo "receiving resources"
echo "-----------------------------------------------"
echo ""
rsync -urv weichen9@alf-clustersrv.mrl.illinois.edu:~/current_work/resources ${PROJECT_DIR}/MD_simulation_on_alanine_dipeptide/current_work/    # get back resources ALF

echo "-----------------------------------------------"
echo "receiving target"
echo "-----------------------------------------------"
echo ""
rsync -urv weichen9@alf-clustersrv.mrl.illinois.edu:~/current_work/target ${PROJECT_DIR}/MD_simulation_on_alanine_dipeptide/current_work/    # get back target ALF
