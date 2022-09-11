
#####---- PANDAS ----#####
#%%
import pandas as pd
import numpy as np
import numpy as np
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

#%%
#Read the data as .tsv files and merge all chr in one big pd.DataFrame
for chr in range(1, 26):
    if chr == 1:
        data = pd.read_csv('/home/paux/FMP/NN/pandas_data/chr'+str(chr)+'.tsv', sep='\t', header=0)
        data['chr'] = 1
    else:
        _data = pd.read_csv('/home/paux/FMP/NN/pandas_data/chr'+str(chr)+'.tsv', sep='\t', header=0)
        _data['chr'] = chr
        data = pd.concat([data, _data], ignore_index= True)
        del(_data)
        print(chr)

#Target check 
print(Counter(data.clinvar_clnsig))
print(len(data.clinvar_clnsig.unique()))
print(Counter(data.effect))
print(len(data.effect.unique()))
print(data.shape)

#Drop problematic rows in 'gerp_rs' column
data = data.loc[data.gerp_rs != '.', :] 

#Change dtype of 'gerp_rs' to float
data['gerp_rs'] = data['gerp_rs'].apply(float)
print(data.shape) # (1466859, 48): 272 droped variants

#Drop variants with no clinvar_clnsig
data = data[data['clinvar_clnsig'].notnull()]
print(data.shape) #(1456847, 48): 10016 droped variants
    
# Nan replacemnt, if any 
data = data.replace('NaN', np.nan)

#Create 2 list with categorical and numerical columns
numerical = ['freqIntGermline',
                'freqIntGermlineNum',
                'freqIntGermlineDem',
                'gp1_asn_af',
                'gp1_eur_af',
                'gp1_afr_af',
                'gp1_af',
                'gerp_rs',
                'mt',
                'phyloP46way_placental',
                'polyphen2_hvar_score',
                'polyphen2_hvar_score',
                'sift_score',
                'cadd_phred',
                'gnomad_af',
                'gnomad_ac',
                'gnomad_an',
                'gnomad_af_popmax',
                'gnomad_ac_popmax',
                'gnomad_an_popmax',
                'gene_coding']
categorical = ['mutationtaster_pred',
            'polyphen2_hvar_pred',
            'sift_pred',
            'effect',
            'gnomad_filter',
            'effect_impact', 
            'functional_class',
            'transcript_biotype']

#Dissmiss thous with low representation
for sig  in data.clinvar_clnsig.unique():
    if (data.clinvar_clnsig == sig).sum() < 3000:
        data = data[data.clinvar_clnsig != sig]

data.shape # (1456585, 48): 262 droped variants 
#%%
#Function to group labels 
def group_labels(sig):
    if sig == 'P|LP':
        return 'P'
    elif sig == 'B|LB':
        return 'B'
    else:
        return sig

#Group some labels on y
data['y'] = data.clinvar_clnsig.apply(lambda x: group_labels(x))
print('y doneee! ')

#Some plots
(data['y'].value_counts()).plot.bar()

#Create a dataframes with each kind of variable: NOT NECESSARY
identificators = data[['locus', 'alleles', 'rsid']]
num_data = data[numerical]
cat_data = data[categorical]

#Impute continuous missing values with median strategy and scale thaem.
imputer = SimpleImputer(missing_values = np.nan, strategy= 'median')
scaler = StandardScaler()
num_data_im = imputer.fit_transform(num_data)
num_data_std = scaler.fit_transform(num_data_im)
num_data = pd.DataFrame(num_data_std, columns= numerical)

#Instance OHE
ohe = OneHotEncoder()

#Perform the encode
features_array = ohe.fit_transform(cat_data).toarray()
cat_data = pd.DataFrame(features_array, columns= np.concatenate(ohe.categories_))
    
#Label encoding on the target
lab_enc = LabelEncoder()
target_array = lab_enc.fit_transform(data.y.values) #408 NaN
target_data = pd.DataFrame(target_array)
target_data.columns = ['encoded_label']

#Cretae a dictionary with the encode number and the label
target_dict = dict(zip(range(len(target_data.encoded_label.unique().tolist())), lab_enc.classes_))

#Rename columns which are np.nan
new_cols = cat_data.columns.tolist()
acc = 0
for n, item in enumerate(new_cols):
    if pd.isna(new_cols[n]):
        new_cols[n] = 'NaN_'+str(acc)
        acc += 1
    else:
        pass
cat_data.columns = new_cols
print('done encode')

#%%
#Create final data frame
identificators.reset_index(drop=True, inplace=True)
num_data.reset_index(drop=True, inplace=True)
cat_data.reset_index(drop=True, inplace=True)  

#%%
#t_data = pd.concat([num_data, cat_data] , axis = 1) 
#%%
nn_data = pd.concat([identificators, num_data, cat_data], axis = 1)

#%%
#Save the finals DFs
nn_data.to_csv('/home/paux/FMP/NN/nn_data/whole_nn_data.csv')
#%%
target_data.to_csv('/home/paux/FMP/NN/nn_data/target_data.csv')

# %%
