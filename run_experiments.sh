#!/bin/bash
echo "Experiments"
echo "TSFRESH"
python cli.py "./config_exp_10hz_15len_euc_tsfresh.json" --training &> exp_10hz_15len_euc_tsfresh.log
python cli.py "./config_exp_10hz_15len_pca_tsfresh.json" --training &> exp_10hz_15len_pca_tsfresh.log
python cli.py "./config_exp_10hz_30len_euc_tsfresh.json" --training &> exp_10hz_30len_euc_tsfresh.log
python cli.py "./config_exp_10hz_30len_pca_tsfresh.json" --training &> exp_10hz_30len_pca_tsfresh.log
python cli.py "./config_exp_10hz_60len_euc_tsfresh.json" --training &> exp_10hz_60len_euc_tsfresh.log
python cli.py "./config_exp_10hz_60len_pca_tsfresh.json" --training &> exp_10hz_60len_pca_tsfresh.log
python cli.py "./config_exp_10hz_90len_euc_tsfresh.json" --training &> exp_10hz_90len_euc_tsfresh.log
python cli.py "./config_exp_10hz_90len_pca_tsfresh.json" --training &> exp_10hz_90len_pca_tsfresh.log
python cli.py "./config_exp_10hz_90len_pca_tsfresh.json" --training &> exp_10hz_90len_pca_tsfresh.log

python cli.py "./config_exp_1hz_15len_euc_tsfresh.json" --training &> exp_1hz_15len_euc_tsfresh.log
python cli.py "./config_exp_1hz_15len_pca_tsfresh.json" --training &> exp_1hz_15len_pca_tsfresh.log
python cli.py "./config_exp_1hz_30len_euc_tsfresh.json" --training &> exp_1hz_30len_euc_tsfresh.log
python cli.py "./config_exp_1hz_30len_pca_tsfresh.json" --training &> exp_1hz_30len_pca_tsfresh.log
python cli.py "./config_exp_1hz_60len_euc_tsfresh.json" --training &> exp_1hz_60len_euc_tsfresh.log
python cli.py "./config_exp_1hz_60len_pca_tsfresh.json" --training &> exp_1hz_60len_pca_tsfresh.log
python cli.py "./config_exp_1hz_90len_euc_tsfresh.json" --training &> exp_1hz_90len_euc_tsfresh.log
python cli.py "./config_exp_1hz_90len_pca_tsfresh.json" --training &> exp_1hz_90len_pca_tsfresh.log
python cli.py "./config_exp_1hz_90len_pca_tsfresh.json" --training &> exp_1hz_90len_pca_tsfresh.log

#echo "SCRIMP_PP"
#python cli.py "./config_exp_10hz_euc_motif.json" --training &> exp_10hz_euc_motif.log
#python cli.py "./config_exp_10hz_pca_motif.json" --training &> exp_10hz_pca_motif.log
#python cli.py "./config_exp_1hz_euc_motif.json" --training &> exp_1hz_euc_motif.log
#python cli.py "./config_exp_1hz_pca_motif.json" --training &> exp_1hz_pca_motif.log

