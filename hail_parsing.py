#%%
import pandas as pd
import numpy as np
import sklearn as skl
import numpy as np
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import matplotlib.pyplot as plt
import scipy as sp
from collections import Counter

#%%
import hail as hl
hl.init()

#%%
# This code will take the most severe transcripts creating a new  struct field with it.
# Foor loop to save each chr as a .tsv
for i in range(1, 26):

    # Read the annotated matrix table
    mt = hl.read_matrix_table("/home/paux/FMP/clinvar_annotated/variants"+str(i)+".ht")

    #Create a vep consequence dictionary ordered by severity (+ to -)
    conseq = ['transcript_ablation',
                    'splice_acceptor_variant',
                    'splice_donor_variant',
                    'stop_gained',
                    'frameshift_variant',
                    'stop_lost',
                    'start_lost',
                    'transcript_amplification',
                    'inframe_insertion',
                    'inframe_deletion',
                    'missense_variant',
                    'protein_altering_variant',
                    'splice_region_variant',
                    'splice_donor_5th_base_variant',
                    'splice_donor_region_variant',
                    'splice_polypyrimidine_tract_variant',
                    'incomplete_terminal_codon_variant',
                    'start_retained_variant',
                    'stop_retained_variant',
                    'synonymous_variant',
                    'coding_sequence_variant',
                    'mature_miRNA_variant',
                    '5_prime_UTR_variant',
                    '3_prime_UTR_variant',
                    'non_coding_transcript_exon_variant',
                    'intron_variant',
                    'NMD_transcript_variant',
                    'non_coding_transcript_variant',
                    'upstream_gene_variant',
                    'downstream_gene_variant',
                    'TFBS_ablation',
                    'TFBS_amplification',
                    'TF_binding_site_variant',
                    'regulatory_region_ablation',
                    'regulatory_region_amplification',
                    'feature_elongation',
                    'regulatory_region_variant',
                    'feature_truncation',
                    'intergenic_variant']
                    
    # Make it a hail object
    conseq_list = hl.literal(conseq)

    #Create new field with the effects splited by ','
    mt_eff = mt.annotate_rows(effs_effect_split= hl.map(lambda x: x.annotate(effect_split = x.effect.split(',')),
                                mt.effs))
    
    # Maps each list item with a number of severiti using 'conseq_list'
    mt_eff = mt_eff.annotate_rows(consequence_numbers_array  = hl.map(lambda eff_element:(hl.map(lambda y: conseq_list.index(y) ,eff_element.effect_split)),mt_eff.effs_effect_split))
    
    # Compute the min value on each row (min value = more sever effect)
    minimum = hl.min(hl.flatten(mt_eff.consequence_numbers_array))
    
    # Create new column with the most sever consequence number
    mt_eff = mt_eff.annotate_rows(contains_min_sev = hl.map(lambda x: x.contains(minimum), mt_eff.consequence_numbers_array))
     
    #Substract the trancript table where is the most severe effect
    mt_eff_final = mt_eff.annotate_rows(most_severe_effs = mt_eff.effs[mt_eff.contains_min_sev.index(hl.bool
    ('True'))])
    
    # Splits effects again, maps and selects the most severe effect.
    mt_eff_final = mt_eff_final.annotate_rows(effects_splited = mt_eff_final.most_severe_effs.effect.split(','))
    mt_eff_final = mt_eff_final.annotate_rows(effects_splited_conseq = hl.map(lambda x: conseq_list.index(x), mt_eff_final.effects_splited))
    minimum = hl.min(mt_eff_final.effects_splited_conseq)
    mt_eff_final = mt_eff_final.annotate_rows(most_severe_effect = conseq_list[minimum]) 
    
    # Drop the cell struct
    mt_final = mt_eff_final.drop(mt_eff_final.effs)
    
    # Disolve a struct (table in a cell)  cell into independent cells
    mt_final = mt_final.annotate_rows(gene_name = mt_final.most_severe_effs.gene_name,
                                    effect_impact = mt_final.most_severe_effs.effect_impact,
                                    transcript_id = mt_final.most_severe_effs.transcript_id,
                                    effect = mt_final.most_severe_effect,
                                    gene_id = mt_final.most_severe_effs.gene_id,
                                    functional_class = mt_final.most_severe_effs.functional_class,
                                    amino_acid_length = mt_final.most_severe_effs.amino_acid_length,
                                    codon_change = mt_final.most_severe_effs.codon_change,
                                    amino_acid_change = mt_final.most_severe_effs.amino_acid_change,
                                    exon_rank = mt_final.most_severe_effs.exon_rank,
                                    transcript_biotype = mt_final.most_severe_effs.transcript_biotype,
                                    gene_coding = mt_final.most_severe_effs.gene_coding)
    
    # Drop fields won't be used
    mt_final = mt_final.drop(mt_final.effs_effect_split)
    mt_final = mt_final.drop(mt_final.clinvar_filter)
    mt_final = mt_final.drop(mt_final.vep_proc_id)
    mt_final = mt_final.drop(mt_final.info)
    mt_final = mt_final.drop(mt_final.consequence_numbers_array)
    mt_final = mt_final.drop(mt_final.contains_min_sev)
    mt_final = mt_final.drop(mt_final.filters)
    mt_final = mt_final.drop(mt_final.effects_splited)
    mt_final = mt_final.drop(mt_final.effects_splited_conseq)
    mt_final = mt_final.drop(mt_final.most_severe_effect)
    mt_final = mt_final.drop(mt_final.most_severe_effs)
    
    #Convert final mt into a pandas table (make a table from the mt, write a .tsv file and open it
    # on pandas as DataFrame)
    t_final = mt_final.make_table()
    t_final.export('/home/paux/FMP/NN/pandas_data/chr'+str(i)+'.tsv')
    print('chr '+str(i)+' done!')
