from ANN_simulation import *


a=simulation_management(None)
a.monitor_status_and_submit_periodically(num=7, monitor_mode = 'always_wait_for_submit')
