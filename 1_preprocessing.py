
# coding: utf-8

# In[ ]:


# Import stuff
from os.path import join
from pandas import DataFrame

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataSink, FreeSurferSource, SelectFiles
from nipype.algorithms.misc import Gunzip

from nipype.interfaces.freesurfer.preprocess import MRIConvert
from nipype.interfaces.freesurfer.model import Binarize
from nipype.interfaces.freesurfer import FSCommand
from nipype.interfaces.fsl.utils import Reorient2Std, MotionOutliers
from nipype.interfaces.fsl.preprocess import MCFLIRT, SliceTimer, FLIRT
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.fsl.model import GLM
from nipype.algorithms.rapidart import ArtifactDetect

# FSL set up- change default file output type
from nipype.interfaces.fsl import FSLCommand
FSLCommand.set_default_output_type('NIFTI_GZ')

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
#from nipype.interfaces.matlab import MatlabCommand
#MatlabCommand.set_default_paths('~/spm12/toolbox')
#MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# Set study variables
analysis_home = '/home/camachocm2/Analysis/ChEC/fmri_proc'
raw_dir = analysis_home + '/raw'
preproc_dir = analysis_home + '/preproc'
firstlevel_dir = analysis_home + '/subjectlevel'
secondlevel_dir = analysis_home + '/grouplevel'
workflow_dir = analysis_home + '/workflows'

#subject_info = DataFrame.read_csv(analysis_home + '/../misc/subjs.csv')
#subjects_list = subject_info['SubjID'].tolist()
subjects_list = ['pilot002']

# FreeSurfer set up - change SUBJECTS_DIR 
fs_dir = analysis_home + 'freesurfer_dir'
FSCommand.set_default_subjects_dir(fs_dir)

# data collection specs
TR = 0.8 #in seconds
num_slices = 29
slice_direction = 3 #3 = z direction
interleaved = True


# In[ ]:


## Data handling Nodes

# Select subjects list
infosource = Node(IdentityInterface(fields=['subjid']),
                  name='infosource')
infosource.iterables = [('subjid', subjects_list)]

# FreeSurferSource - Data grabber specific for FreeSurfer data
fssource = Node(FreeSurferSource(subjects_dir=fs_dir),
                run_without_submitting=True,
                name='fssource')

# Pull files
file_template = {'chec_ap': raw_dir + '/chec_2_ap.nii.gz', 
                 'chec_pa': raw_dir + '/chec_1_pa.nii.gz', 
                 'func': raw_dir + '/chec_movie_PAec.nii.gz'}
selectfiles = Node(SelectFiles(file_template), name='selectfiles')

# Sink data of interest (mostly for QC)
substitutions = [('_subjid_', '')] #output file name substitutions
datasink = Node(DataSink(base_directory = output_dir,
                        container = output_dir,
                        substitutions = substitutions), 
                name='datasink')


# In[ ]:


def extract_dicom_info(i,dicom, dicoms_info, volume_name):    
    from pydicom import dcmread
    from pandas import DataFrame
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    info = dcmread(dicom)
    etl = info[0x18,0x91].value
    te = info[0x18,0x81].value
    fa = info[0x181314].value
    tr = info[0x18, 0x80].value
    pid = info[0x10, 0x20].value
    slice_thick = info[0x18, 0x50].value
    pixel_size = info[0x28, 0x30].value
    acq_matrix = info[0x181310].value
    slice_timing = info[0x191029].value
    acquisition_num = info[0x20, 0x12].value

    dicoms_info.loc[i] = [pid, etl, te, fa, tr, acquisition_num, 
                               slice_thick, pixel_size, acq_matrix, slice_timing, volume_name]
    return(dicoms_info)

def make_nifti(dicom, output_dir, volume_name):
    from pandas import DataFrame
    from os.path import basename
    import shutil
    from nipype.interfaces.freesurfer.preprocess import MRIConvert
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    temp_path = output_dir + '/' + basename(dicom)
    
    shutil.move(dicom,temp_path)
    mrc = MRIConvert()
    mrc.inputs.in_file = temp_path
    mrc.inputs.out_file = volume_name
    mrc.run()
    shutil.move(temp_path, dicom)
    return()


# In[ ]:


from glob import glob
from pandas import DataFrame

dicoms_folder = '/home/camachocm2/Analysis/ChEC/fmri_proc/raw/pilot002/chec_movie_PA'
output_dir = '/home/camachocm2/Analysis/ChEC/fmri_proc/raw/pilot002/chec_sort'

files = glob(dicoms_folder + '/*')
files = sorted(files)
dicoms_info = DataFrame(columns = ['PatientID','EchoTrainLength','TE','FlipAngle','TR', 'AcqNumber',
                                   'SliceThickness','PixelSize','AcqMatrix','SliceTiming','VolumeName'])

for volnum in range(1300,1600):
    volume_name = output_dir + '/vol' + str(volnum).zfill(5) + '.nii.gz'
    
    dicom_info = extract_dicom_info(volnum, files[volnum], dicoms_info, volume_name)
    make_nifti(files[volnum], output_dir, volume_name)
    
dicom_info.head()


# In[ ]:


## fMRI Data processing nodes

# get info from ME dicoms

# sort echos and make niftis

# unwarp fmri


# reorient images to MNI space standard
reorientFunc = Node(Reorient2Std(terminal_output='file'),
                   name='reorientFunc')

# perform slice time correction
slicetime = Node(SliceTimer(index_dir=False,
                           interleaved=True,
                           time_repetition=2),
                name='slicetime')

# realignment using mcflirt
realignmc = Node(MCFLIRT(save_plots=True),
                name='realignmc')

# Coregistration using flirt
coregflt = Node(FLIRT(),
               name='coregflt')
coregflt2 = Node(FLIRT(apply_xfm=True, 
                       out_file='preproc_func.nii'),
                name='coregflt2')

#unzip file before feeding into ART
gunzip = Node(Gunzip(), name='gunzip')

# Artifact detection for scrubbing/motion assessment
art = Node(ArtifactDetect(mask_type='file',
                          parameter_source='FSL',
                          bound_by_brainmask=True,
                          norm_threshold=1,
                          zintensity_threshold=3,
                          use_differences=[True, False]),
           name='art')


# In[ ]:


## Anatomical processing
# Convert skullstripped brain to nii, resample to 2mm^3
resample = Node(MRIConvert(out_type='nii',
                          vox_size=(2,2,2)),
               name='resample')

# Reorient anat to MNI space
reorientAnat = Node(Reorient2Std(terminal_output='file'),
                   name='reorientAnat')

# Create binary mask of resampled, skullstripped anat, dilate, and erode to fill gaps
binarize = Node(Binarize(dilate=1,
                        erode=1,
                        invert=False,
                        min=1,
                        max=255),
               name='binarize')


# In[ ]:


# Data QC nodes
def create_coreg_plot(epi,anat):
    import os
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nilearn import plotting
    
    coreg_filename='coregistration.png'
    display = plotting.plot_anat(epi, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'coregistration to anatomy')
    display.add_edges(anat)
    display.savefig(coreg_filename) 
    display.close()
    coreg_file = os.path.abspath(coreg_filename)
    
    return(coreg_file)

def check_mask_coverage(epi,brainmask):
    import os
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nilearn import plotting
    
    maskcheck_filename='maskcheck.png'
    display = plotting.plot_anat(epi, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'brainmask coverage')
    display.add_contours(brainmask,levels=[.5], colors='r')
    display.savefig(maskcheck_filename)
    display.close()
    maskcheck_file = os.path.abspath(maskcheck_filename)

    return(maskcheck_file)

make_coreg_img = Node(name='make_coreg_img',
                      interface=Function(input_names=['epi','anat'],
                                         output_names=['coreg_file'],
                                         function=create_coreg_plot))

make_checkmask_img = Node(name='make_checkmask_img',
                      interface=Function(input_names=['epi','brainmask'],
                                         output_names=['maskcheck_file'],
                                         function=check_mask_coverage))


# In[ ]:


preprocflow = Workflow(name='preprocflow')

preprocflow.connect([(infosource, selectfunc, [('subjid','subjid')]),
                     (selectfunc, convertdicoms,[('dicom_file','in_file')]),
                     (convertdicoms,reorientFunc, [('out_file','in_file')]),
                     (infosource, fsid, [('subjid','subjid')]), 
                     (fsid, fssource, [('fs_id','subject_id')]),
                     (fssource, resample, [('brainmask','in_file')]),
                     (resample, reorientAnat, [('out_file','in_file')]),
                     (reorientAnat, binarize, [('out_file','in_file')]),
                     (reorientFunc, slicetime, [('out_file','in_file')]),
                     (slicetime, realignmc, [('slice_time_corrected_file','in_file')]),
                     (reorientAnat, coregflt, [('out_file','reference')]),
                     (realignmc, coregflt, [('out_file','in_file')]),
                     (realignmc, coregflt2, [('out_file','in_file')]),
                     (coregflt, coregflt2, [('out_matrix_file','in_matrix_file')]),
                     (reorientAnat, coregflt2, [('out_file','reference')]),
                     (binarize, art, [('binary_file','mask_file')]),
                     (coregflt2, art, [('out_file','realigned_files')]),
                     (realignmc, art, [('par_file','realignment_parameters')]),
                     (coregflt, make_coreg_img, [('out_file','epi')]),
                     (reorientAnat, make_coreg_img, [('out_file','anat')]),
                     (coregflt, make_checkmask_img, [('out_file','epi')]),
                     (binarize, make_checkmask_img, [('binary_file','brainmask')]),
                     
                     (reorientAnat, datasink, [('out_file','reoriented_anat')]),
                     (binarize, datasink, [('binary_file','binarized_anat')]),
                     (make_checkmask_img, datasink, [('maskcheck_file','WM_checkmask_image')]),
                     (make_coreg_img, datasink, [('coreg_file','WM_coreg_image')]),
                     (coregflt2, datasink, [('out_file','WM_coreg_func')]),
                     (coregflt, datasink, [('out_file','WM_coreg_firstvol')]),
                     (art, datasink, [('plot_files','WM_art_plot')]), 
                     (art, datasink, [('outlier_files','WM_art_outliers')]),
                     (realignmc, datasink, [('par_file','WM_mcflirt_displacement')])
                    ])
preprocflow.base_dir = wkflow_dir
preprocflow.write_graph(graph2use='flat')
preprocflow.run('MultiProc', plugin_args={'n_procs': 4, 'memory_gb':10})

