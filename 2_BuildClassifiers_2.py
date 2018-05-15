
# coding: utf-8

# # Build the Classifier
# This notebook is dedicated to creating classifiers and run classification analyses of interest on neuroimaging data.
# 
# Can we accurately classify:
# - adults vs. children
# - condition within adults
# - condition within children

# In[1]:


from pandas import DataFrame, Series, read_csv

# Study specific variables
study_home = '/home/camachocm2/Analysis/KidVid_MVPA'
standard_mask = study_home + '/template/MNI152_T1_2mm_brain_mask_KV.nii.gz'
template = study_home + '/template/MNI152_T1_1mm_brain.nii.gz'
sub_data_file = study_home + '/doc/subjectinfo.csv'
preproc_dir = study_home + '/analysis/preproc/betas'
output_dir = study_home + '/analysis/classifier'

condition_data = read_csv(study_home + '/doc/conditionslist.csv')
subject_info = read_csv(sub_data_file)


# In[2]:


## Create a conditions list for the feature set
condition_labels = condition_data['labels'].tolist()
subjects_list = subject_info['subjID'].tolist()
age_group_list = subject_info['group'].tolist()

conditions = condition_data
conditions['subject'] = Series(subjects_list[0], index=conditions.index)
conditions['ageGroup'] = Series(age_group_list[0], index=conditions.index)

for a in range(1,len(subjects_list)):
    temp=DataFrame()
    temp['labels'] = Series(condition_labels)
    temp['subject'] = Series(subjects_list[a], index=temp.index)
    temp['ageGroup'] = Series(age_group_list[a], index=temp.index)
    conditions = conditions.append(temp, ignore_index=True)

#conditions.to_csv(output_dir + '/featureset_key.csv')
conditions.describe()


# In[3]:


## Temporally concatenate all the parameter estimates from preproc to create a feature set
from glob import glob
from nipype.interfaces.fsl.utils import Merge
files = glob(preproc_dir + '/*/betas.nii.gz')
files = sorted(files)

bold_feature_data = output_dir + '/featureset.nii.gz'

merge = Merge()
merge.inputs.in_files = files
merge.inputs.dimension = 't'
merge.inputs.merged_file = bold_feature_data
#merge.run()


# In[4]:


# determine which analysis to run
analysis = 'neutral'

if analysis == 'all_conditions':
    mask = conditions['labels'].isin(['negative','positive','neutral'])
    labels = conditions['labels']
elif analysis == 'adults':
    mask = conditions['ageGroup'].isin(['adult'])
    labels = conditions['labels']
elif analysis == 'children':
    mask = conditions['ageGroup'].isin(['child'])
    labels = conditions['labels']
elif analysis == 'allConds_predAge':
    mask = conditions['labels'].isin(['negative','positive','neutral'])
    labels = conditions['ageGroup']
elif analysis == 'negative':
    mask = conditions['labels'].isin(['negative'])
    labels = conditions['ageGroup']
elif analysis == 'positive':
    mask = conditions['labels'].isin(['positive'])
    labels = conditions['ageGroup']
elif analysis == 'neutral':
    mask = conditions['labels'].isin(['neutral'])
    labels = conditions['ageGroup']

results_file = open(output_dir + '/results_' + analysis + '.txt','w')
conditions[mask].describe()


# ## Perform the actual support vector classification

# In[5]:


# Perform the support vector classification
from nilearn.input_data import NiftiMasker
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline

# Set up the support vector classifier
svc = SVC(kernel='linear')
masker = NiftiMasker(mask_img=standard_mask,standardize=True, 
                     memory='nilearn_cache', memory_level=1)
feature_selection = SelectPercentile(f_classif, percentile=5)
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

# Run the classifier
X = masker.fit_transform(bold_feature_data)
X = X[mask]
maskedlabels=labels[mask]
anova_svc.fit(X, maskedlabels)
y_pred = anova_svc.predict(X)

# Obtain prediction values via cross validation
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score

cv = LeaveOneLabelOut(conditions['subject'][mask])
cv_scores = cross_val_score(anova_svc, X, maskedlabels, cv=cv)
classification_accuracy = cv_scores.mean()

print("this is for analysis: " + analysis)

print("Classification accuracy: %.4f / Chance level: %f" % 
      (classification_accuracy, 1. / len(labels.unique())))
results_file.write("Classification accuracy: %.4f / Chance level: %f" % (classification_accuracy, 1. / len(labels.unique())))

# ## Perform permutation testing to get a p-value for the classifier

# In[ ]:


from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt

# Perform permutation testing to get a p-value
score, permutation_scores, pvalue = permutation_test_score(svc, X, y_pred, scoring="accuracy", 
                                                           cv=cv, n_permutations=500, n_jobs=12)

print("Classification score %s (pvalue : %s)" % (score, pvalue))

plt.hist(permutation_scores, 20, label='Permutation scores',
         edgecolor='black')
ylim = plt.ylim()
plt.plot(2 * [score], ylim, '--g', linewidth=3,
         label='Classification Score'
         ' (pvalue %s)' % pvalue)
plt.plot(2 * [1. / len(labels.unique())], ylim, '--k', linewidth=3, label='Luck')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
plt.savefig(output_dir + '/permutation_plot_' + analysis + '.png', transparent=True)

results_file.write("Classification score %s (pvalue : %s)" % (score, pvalue))
results_file.close()

import shelve

filename = output_dir + '/' + analysis + '_shelved.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()
