# Python_ASAP
#PA - python for degradation rate analysis
#Github guide: https://guides.github.com/activities/hello-world/

#Initial excel file read in: https://www.youtube.com/watch?v=FXhED53VZ50
#import modules
Import numpy as np
Import pandas as pd
#load data file
ASAP_data = pd.read_csv('file_name.csv')
time = ASAP_data['column_header0']
percent_degradation = ASAP_data ['column_header1']
temperature = ASAP_data ['column_header2']
Relative_humidity = ASAP_data ['column_header3']

t0_normalized_degradation = percent_degradation - percent_degradation[0]
#Unsure how to index cells if t=0 (more than one t0 sample)
t0_normalized_lndeg = np.log(percent_degradation[1:])
