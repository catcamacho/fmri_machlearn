
# coding: utf-8

# # Preprocessing for machine learning
# This notebook is designed to preprocess  neuroimaging and behavioral data for machine learning analyses. This includes mean-centering and normalizing vector data (i.e. questionnaire scores, demographics) and extracting a beta series from the fMRI data. The beta series is a result of deconvolving the time series for each trial which allows us to use the beta maps as entries for classification rather than a BOLD timeseries (see Mumford 2012 for a more thorough explanation of this; these steps are also outlined below).

# In[1]:


from pandas import DataFrame, Series, read_csv

# Study specific variables
study_home = '/home/camachocm2/Analysis/KidVid_MVPA'
standard_mask = '/home/camachocm2/Analysis/Templates/MNI152_T1_2mm_brain_mask.nii.gz'

subject_info = read_csv(study_home + '/doc/subjectinfo.csv')
subjects_list = subject_info['subjID'].tolist()
version = subject_info['version'].tolist()


# ### fMRI data preprocessing

# In[2]:


from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.fsl.utils import Merge, ImageMeants, Split, MotionOutliers
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.nipy.model import FitGLM, EstimateContrast
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.interfaces.fsl.model import GLM, Level1Design, FEATModel
from nipype.interfaces.fsl.maths import ApplyMask

preproc_fmri = study_home + '/processed_data'
output_dir = study_home + '/analysis/preproc'
timing_dir = study_home + '/timing'
workflow_dir = study_home + '/workflows'

TR=2 #in seconds

# FSL set up- change default file output type
from nipype.interfaces.fsl import FSLCommand
FSLCommand.set_default_output_type('NIFTI_GZ')


# In[3]:


# Data handling nodes
infosource = Node(IdentityInterface(fields=['subjid','version']), 
                  name='infosource')
infosource.iterables = [('subjid', subjects_list),('version', version)]
infosource.synchronize = True

template = {'proc_func': preproc_fmri + '/{subjid}/preproc_func.nii.gz', 
            'raw_func': preproc_fmri + '/{subjid}/raw_func.nii.gz', 
            'timing':timing_dir + '/{version}.txt'}
selectfiles = Node(SelectFiles(template), name='selectfiles')

substitutions = [('_subjid_', ''),
                 ('_version_version1',''), 
                 ('_version_version2',''), 
                 ('_version_version3','')]
datasink = Node(DataSink(substitutions=substitutions, 
                         base_directory=output_dir,
                         container=output_dir), 
                name='datasink')


# In[4]:


# Extract timing for Beta Series Method- mark trials as high and low motion
def pull_timing(timing_file,motion_file):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from pandas import DataFrame,Series,read_table
    from nipype.interfaces.base import Bunch
    
    timing = read_table(timing_file)
    motion_series = read_table(motion_file, header=None, names=['fd'])
    motion = motion_series['fd'].tolist()
    
    names = timing['Condition'].tolist()
    cond_names = [names[a] + str(a) for a in range(0,len(names))]
    print(cond_names)
    onsets = []
    for a in timing['Onset'].tolist():
        onsets.append([a])

    durations = []
    for b in timing['Duration'].tolist():
        durations.append([b])
    
    #make bunch file
    timing_bunch = []
    timing_bunch.insert(0,Bunch(conditions=cond_names,
                                onsets=onsets,
                                durations=durations,
                                amplitudes=None,
                                tmod=None,
                                pmod=None,
                                regressor_names=['fd'],
                                regressors=[motion]))
    return(timing_bunch)

# Function to create contrast lists from a bunch file
def beta_contrasts(timing_bunch):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    from nipype.interfaces.base import Bunch
    from numpy import zeros
    
    conditions_names = timing_bunch[0].conditions
    
    # Make the contrast vectors for each trial
    boolean_con_lists = []
    num_cons = len(conditions_names)
    for i in range(0,num_cons):
        boo = zeros(num_cons)
        boo[i] = 1
        boolean_con_lists.append(list(boo))
    
    # Create the list of lists for the full contrast info
    contrasts_list = []
    for a in range(0,num_cons):
        con = (conditions_names[a], 'T', conditions_names, boolean_con_lists[a])
        contrasts_list.append(con)
    
    return(contrasts_list)


# In[5]:


# Get framewise displacement to use as a regressor in the GLM
get_fd = Node(MotionOutliers(metric='fd',
                             out_metric_values='FD.txt', 
                             out_metric_plot='FD.png'),
              name='get_fd')

# Extract timing
pull_timing = Node(Function(input_names=['timing_file','motion_file'],
                            output_names=['timing_bunch'],
                            function=pull_timing), 
                   name='pull_timing')

# create the list of T-contrasts
define_contrasts = Node(Function(input_names=['timing_bunch'], 
                                 output_names = ['contrasts_list'], 
                                 function=beta_contrasts),
                        name = 'define_contrasts')

# Specify FSL model - input bunch file called subject_info
modelspec = Node(SpecifyModel(time_repetition=TR, 
                              input_units='secs',
                              high_pass_filter_cutoff=128),
                 name='modelspec')

# Generate a level 1 design
level1design = Node(Level1Design(bases={'dgamma':{'derivs': False}},
                                 interscan_interval=TR, 
                                 model_serial_correlations=True), 
                    name='level1design')

# Estimate Level 1
generate_model = Node(FEATModel(), 
                      name='generate_model')

# Run GLM
extract_pes = Node(GLM(out_file = 'betas.nii.gz'), 
                   name='extract_pes')

# In[ ]:


preprocflow = Workflow(name='preprocflow')
preprocflow.connect([(infosource, selectfiles,[('subjid','subjid')]),
                     (infosource, selectfiles,[('version','version')]),
                     (selectfiles, get_fd, [('raw_func','in_file')]),
                     (selectfiles, pull_timing, [('timing','timing_file')]),
                     (selectfiles, modelspec, [('proc_func','functional_runs')]),
                     (get_fd, pull_timing, [('out_metric_values','motion_file')]),
                     (pull_timing, modelspec, [('timing_bunch','subject_info')]),
                     (pull_timing, define_contrasts, [('timing_bunch','timing_bunch')]),
                     (define_contrasts, level1design, [('contrasts_list','contrasts')]),
                     (modelspec, level1design, [('session_info','session_info')]),
                     (level1design,generate_model, [('ev_files','ev_files')]),
                     (level1design,generate_model, [('fsf_files','fsf_file')]),
                     (generate_model,extract_pes, [('design_file','design')]),
                     (generate_model,extract_pes, [('con_file','contrasts')]),
                     (selectfiles,extract_pes, [('proc_func','in_file')]),
                     
                     (extract_pes,datasink,[('out_file','betas')]),
                     (get_fd, datasink, [('out_metric_values','fd_motion')]),
                     (get_fd, datasink, [('out_metric_plot','fd_motion_plots')]),
                     (generate_model,datasink,[('design_image','design_image')])
                    ])
preprocflow.base_dir = workflow_dir
preprocflow.write_graph(graph2use='flat')
preprocflow.run('MultiProc', plugin_args={'n_procs': 4,'memory_gb':10})

