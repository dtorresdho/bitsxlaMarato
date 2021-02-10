#!/usr/bin/python3.7

import logging 
logging.disable(logging.NOTSET)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Importing libraries
import re
import cgi
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pathlib
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from bs4.dammit import EntitySubstitution

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity


def find_cluster_newcase(case_id = None, case_desc = None):
    case_nb = 2501
    ### Read the case
    cases_new = {}
    if case_desc == None:
        doc = f"data/covid-marato-clinical-cases/covid_marato_{case_id}.txt"
        with open(doc, 'r') as reader:
            case_id = int(case_id)
            cases_new[case_id] = re.sub('\\n', ' ', reader.read().lower())
    else:
        case_id = 9999
        cases_new[case_id] = re.sub('\\n', ' ', case_desc.lower())

    X_new = pd.DataFrame.from_dict(cases_new, orient='index', columns=['case'])

    ### Encode the case with saved tokenizer


    # loading Tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # encode
    seqs = tokenizer.texts_to_sequences(X_new['case'].values)
    # pad
    maxlen = 900
    pad_seqs = pad_sequences(seqs, maxlen)
    #print("original input shape:", pad_seqs.shape)

    # reshape  input
    timesteps = 1
    dataX = []
    for i in range(len(pad_seqs)):
        x = pad_seqs[i:(i+timesteps), :]
        dataX.append(x)
    x_t_rs = np.array(dataX)
    #print("reshaped:", x_t_rs.shape)

    ### get the representation of the case with enncoder
    encoder = keras.models.load_model('encoder_covid.h5', compile=False)
    latent_rep_x_new = encoder.predict(x_t_rs)[0].tolist()
    #print("Latent representation dimension:",np.shape(latent_rep_x_new))

    #print("...\n")

    ### get the distance matrix between cases in the latent space
    # load representation of the cases in corpus
    latent_space_df_with_x_new = pd.read_csv("ae_lstm_latent_space_with_cluster.csv",index_col=0)
    # append new case
    latent_rep_x_new.append(np.nan)
    latent_space_df_with_x_new.loc[case_nb,:] = latent_rep_x_new

    # compute distance matrix
    dist = 1- cosine_similarity(latent_space_df_with_x_new.drop('cluster',axis=1))

    dist_out = pd.DataFrame(data=dist,
                          index=latent_space_df_with_x_new.index,
                          columns=latent_space_df_with_x_new.index)
    # add cluster to distance matrix
    dist_out['cluster'] = latent_space_df_with_x_new['cluster']

    # compute the mean distance between new case and corpus cases in each cluster
    clust_nb = []
    mean_pairw_dist_clust = []
    for i in range(0,int(max(dist_out['cluster']))+1):
        dist_mat_clust = dist_out.loc[dist_out['cluster']==i,case_nb]
        clust_nb.append(i)
        mean_pairw_dist_clust.append(dist_mat_clust.mean())

    doc_simil_clust = pd.DataFrame({'cluster':clust_nb,'mean_pairwdist':mean_pairw_dist_clust}).sort_values('mean_pairwdist').reset_index(drop=True)
    #print("Top 3 clusters:\n",doc_simil_clust.head(3))
    top_cluster = doc_simil_clust['cluster'][0]

    ### get dominant topics
    topics_all_clusters = pd.read_csv("clusters_topics.csv")
    topics_top_cluster = topics_all_clusters[topics_all_clusters['cluster']== top_cluster]

    return case_id, cases_new[case_id], doc_simil_clust, top_cluster, topics_all_clusters, topics_top_cluster




def find_keyword(keyword):
    #case_id, case_desc, doc_simil_clust, top_cluster, topics_all_clusters, topics_top_cluster = find_cluster_newcase(case_id=1, case_desc=None)
    topics_all_clusters = pd.read_csv("clusters_topics.csv")
    clusters = []
    for index, row in topics_all_clusters.iterrows():
        x = row['topics'][1:-1].replace("'",'').split(' ')
        if len(list(filter(re.compile(f".*{keyword}.*").match, x))) > 0:
            clusters.append(row)

    return clusters, topics_all_clusters





def main():
    escaper = EntitySubstitution()
    form = cgi.FieldStorage()
    print("Content-type: text/html\n\n")

    print('''
                <html>
                <head>
                <title>BitsxlaMarato 2020 - La FrancoArgentina Team</title>
                <meta http-equiv="Content-Type" content="text/html; charset=utf-8">

                <script type="text/javascript" src="/jquery/jquery-3.3.1.min.js"></script>

                <link rel="stylesheet" type="text/css" href="/jquery/DataTables/bootstrap.min.css"/>
                <link href="/jquery/DataTables/DataTables-1.10.18/css/jquery.dataTables.css" rel="stylesheet" type="text/css" />
                <script src="/jquery/DataTables/DataTables-1.10.18/js/jquery.dataTables.js"></script>
                <link rel="stylesheet" type="text/css" href="/jquery/DataTables/dataTables.bootstrap.min.css"/>
                <script src="stylesheet" type="text/css" href="/jquery/DataTables/dataTables.js"/></script>
                
		<style>

    			.blue-button {
        			display: inline-block;
        			-webkit-box-sizing: content-box;
        			-moz-box-sizing: content-box;
        			box-sizing: content-box;
        			cursor: pointer;
        			padding: 5px 15px;
        			border: 1px solid #018dc4;
        			-webkit-border-radius: 3px;
        			border-radius: 3px;
        			font: normal 16px/normal "Times New Roman", Times, serif;
        			color: rgba(255,255,255,0.9);
        			-o-text-overflow: clip;
        			text-overflow: clip;
        			background: #787A7D;
        			-webkit-box-shadow: 3px 3px 5px 0 rgba(0,0,0,0.2) ;
        			box-shadow: 3px 3px 5px 0 rgba(0,0,0,0.2) ;
        			text-shadow: -1px -1px 0 rgba(15,73,168,0.66) ;
        			-webkit-transition: all 300ms cubic-bezier(0.42, 0, 0.58, 1);
        			-moz-transition: all 300ms cubic-bezier(0.42, 0, 0.58, 1);
        			-o-transition: all 300ms cubic-bezier(0.42, 0, 0.58, 1);
        			transition: all 300ms cubic-bezier(0.42, 0, 0.58, 1);
    			}

    			.light-blue-button {
        			display: inline-block;
        			-webkit-box-sizing: content-box;
        			-moz-box-sizing: content-box;
        			box-sizing: content-box;
        			cursor: pointer;
        			padding: 2px 8px;
        			border: 1px solid #018dc4;
        			-webkit-border-radius: 3px;
        			border-radius: 3px;
        			font: normal 12px/normal "Times New Roman", Times, serif;
        			color: rgba(255,255,255,0.9);
        			-o-text-overflow: clip;
        			text-overflow: clip;
        			background: #a6cfe0;
        			-webkit-box-shadow: 3px 3px 5px 0 rgba(0,0,0,0.2) ;
        			box-shadow: 3px 3px 5px 0 rgba(0,0,0,0.2) ;
        			text-shadow: -1px -1px 0 rgba(15,73,168,0.66) ;
        			-webkit-transition: all 300ms cubic-bezier(0.42, 0, 0.58, 1);
        			-moz-transition: all 300ms cubic-bezier(0.42, 0, 0.58, 1);
        			-o-transition: all 300ms cubic-bezier(0.42, 0, 0.58, 1);
        			transition: all 300ms cubic-bezier(0.42, 0, 0.58, 1);
    			}

    			.text-input {
        			display: inline-block;
        			-webkit-box-sizing: content-box;
        			-moz-box-sizing: content-box;
        			box-sizing: content-box;
        			padding: 4px 10px;
        			border: 1px solid #b7b7b7;
					margin-bottom: 30px;
        			-webkit-border-radius: 3px;
        			border-radius: 3px;
        			font: normal 16px/normal "Times New Roman", Times, serif;
        			color: rgba(0,142,198,1);
        			-o-text-overflow: clip;
        			text-overflow: clip;
        			letter-spacing: 1px;
        			word-spacing: 2px;
        			background: rgba(234,234,234,1);
        			-webkit-box-shadow: 2px 2px 2px 0 rgba(0,0,0,0.2) inset;
        			box-shadow: 2px 2px 2px 0 rgba(0,0,0,0.2) inset;
        			text-shadow: 1px 1px 0 rgba(255,255,255,0.66) ;
        			-webkit-transition: all 200ms cubic-bezier(0.42, 0, 0.58, 1);
        			-moz-transition: all 200ms cubic-bezier(0.42, 0, 0.58, 1);
        			-o-transition: all 200ms cubic-bezier(0.42, 0, 0.58, 1);
        			transition: all 200ms cubic-bezier(0.42, 0, 0.58, 1);
    			}

			#title-1 {
				font-family: Verdana, Geneva, sans-serif;
				font-size: 24px;
				letter-spacing: 0.4px;
				word-spacing: 0px;
				color: #000000;
				font-weight: 700;
				text-decoration: none;
				font-style: normal;
				font-variant: normal;
				text-transform: none;
			}

			#title-2 {
				font-family: Verdana, Geneva, sans-serif;
				font-size: 20px;
				letter-spacing: 0.4px;
				word-spacing: 0px;
				color: #000000;
				font-weight: 700;
				text-decoration: none;
				font-style: normal;
				font-variant: normal;
				text-transform: none;
				vertical-align: middle;
				text-align: center;
			}

			#title-3 {
				font-family: Verdana, Geneva, sans-serif;
				font-size: 12px;
				letter-spacing: 0.4px;
				word-spacing: 0px;
				color: #000000;
				font-weight: 700;
				text-decoration: none;
				font-style: normal;
				font-variant: normal;
				text-transform: none;
			}

			#all-content {
				margin: auto;
			}

			.center {
				text-align: center;
			}

			.row {
				min-height: 100px;
				position: relative;
				text-align: center;
			}

			.column_center {
  				display: inline-block;
  				padding: 20px;
  				border:1px solid red;
			}

			label {
  				float: center;
  				margin: 10 30px;
			}


		</style>

                </head>


    ''')
    print('''<body>
            <div id="all-content">
            <div class="row">
            <div id="title-2">BitsxlaMarato 2020 - La FrancoArgentina Team</div>
            <br>
            <br>
            ''')

    if form.getfirst('action_on_post', None) == "clinical_description":
        case_id, case_desc, doc_simil_clust, top_cluster, topics_all_clusters, topics_top_cluster = find_cluster_newcase(case_id=None, case_desc=form.getfirst('clinical_desc'))
        abc = 1
    elif form.getfirst('action_on_post', None) == "case_id":
        case_id, case_desc, doc_simil_clust, top_cluster, topics_all_clusters, topics_top_cluster = find_cluster_newcase(case_id=form.getfirst('clinical_id'), case_desc=None)
        abc = 1
    elif form.getfirst('action_on_post', None) == "keyword":
        clusters, topics_all_clusters = find_keyword(keyword=form.getfirst('keyword'))
        abc = 2
    else:
        print(''' Nothing to do ''')

    if abc == 1:
        print("<table border=0>")

        print(f"<tr style='text-align:left'>")
        print("<td style='width:20%; vertical-align:top'>")
        print("<label>")
        print(f"<b>Case ID:</b>")
        print("</label>")
        print("</td>")
        print("<td style='width:85%'>")
        print(f"<b>{str(case_id)}")
        print("</td>")
        print("</tr>")
    
        print(f"<tr style='text-align:left'>")
        print("<td style='width:20%; vertical-align:top'>")
        print("<label>")
        print(f"<b>Case description:</b>")
        print("</label>")
        print("</td>")
        print("<td style='width:85%'>")
        print(f"{escaper.substitute_html(case_desc)}")
        print("</td>")
        print("</tr>")
    
        print(f"<tr style='text-align:left'>")
        print("<td style='width:20%; vertical-align:top'>")
        print("<label>")
        print(f"<b>Assigned to cluster:</b>")
        print("</label>")
        print("</td>")
        print("<td style='width:85%'>")
        print(f"{escaper.substitute_html(str(top_cluster))}")
        print("</td>")
        print("</tr>")

        print(f"<tr style='text-align:left'>")
        print("<td style='width:20%; vertical-align:top'>")
        print("<label>")
        print("<b>Topics in the assigned cluster:</b>")
        print("</label>")
        print("</td>")
        print("<td style='width:85%'>")
        for index, row in topics_top_cluster.iterrows():
            print(f"{escaper.substitute_html(str(row[1]))}<br>")
        print(f"</td>")
        print(f"</tr>")
    
        print(f"<tr style='text-align:left'>")
        print("<td style='width:20%; vertical-align:top'>")
        print("<label>")
        print("<b>Mean pairwise distance to each cluster:</b>")
        print("</label>")
        print("</td>")
        print("<td>")
        print("<table border='0'>")
        print("<tr>")
        print("<td style='width:45%; text-align:center'><b>Cluster</b></td>")
        print("<td style='width:55%; text-align:center'><b>Distance</b></td>")
        print("</tr>")
        for index, row in doc_simil_clust.iterrows():
            print("<tr>")
            print(f"<td style='width:45%; text-align:center'><b>{escaper.substitute_html(str(int(row[0])))}</b></td>")
            print(f"<td style='width:55%; text-align:right'>{escaper.substitute_html(str(row[1]))}</td>")
            print(f"</tr>")
        print("</table>")
        print("</td>")
        print("</table>")
        print("<hr>")

        print("<table border=0>")
        print("<tr style='text-align:left'>")
        print("<td style='width:20%; vertical-align:top'>")
        print("<label>")
        print("<b>Topics in all clusters:</caption>")
        print("</label>")
        print("</td>")
        print("<td>")
        print("<table border='0'>")
        print("<tr>")
        print("<th style='text-align:center'><b>Cluster</b></th>")
        print("<th style='text-align:center'><b>Topics</b></th>")
        print("</tr>")
        a = None
        for index, row in topics_all_clusters.iterrows():
            if (a is None):
                print(f"<tr>")
                print(f"<td style='text-align:center'>{escaper.substitute_html(str(row[0]))}</td>")
                print("<td>")
                print(f"{escaper.substitute_html(str(row[1]))}<br>")
                a = row[0]
            if (a != row[0]):
                print("</td>")
                print(f"</tr>")
                print("<tr><td colspan=2><hr></td></tr>")
                print(f"<tr>")
                print(f"<td style='text-align:center'>{escaper.substitute_html(str(row[0]))}</td>")
                print("<td>")
                print(f"{escaper.substitute_html(str(row[1]))}<br>")
                a = row[0]
            else:
                print(f"{escaper.substitute_html(str(row[1]))}<br>")
        print("</td>")
        print(f"</tr>")
        print("</table>")
        print("</td>")
        print("</table>")
        print("<br>")
    else:
        #DTD
        #print(f"Clusters: {clusters.items()}")
        if len(clusters) > 0:
            print("<center><h2>Keyword found in the following clusters:</h2></center>")
            print("<center>")
            print("<table border=0><tr><th calss='text-center'>Cluster</th><th class='text-center'>Topics</th></tr>")
            for row in clusters:
                print(f"<tr><td align='center'><b>{row['cluster']}</b></td><td>{row['topics']}</td></tr>")
            print("</table>")
            print("<center><h3>Look for these clusters below in order to find all the words in each of them.</h3></center>")
            print("</center>")
        else:
            print("<center><h2>Keyword not found in any cluster</h2></center>")
        print("<br><hr>")
        print("<center>")
        print("<b>Topics in all clusters:</caption>")
        print("</label>")
        print("</td>")
        print("<td>")
        print("<table border='0'>")
        print("<tr>")
        print("<th style='text-align:center'><b>Cluster</b></th>")
        print("<th style='text-align:center'><b>Topics</b></th>")
        print("</tr>")
        a = None
        b = None
        #print("<center><h2>All clusters in the model:</h2></center>")
        for index, row in topics_all_clusters.iterrows():
            if (a is None) & (b is None):
                print(f"<tr>")
                print(f"<td style='text-align:center'><b>{escaper.substitute_html(str(row[0]))}</b></td>")
                print("<td>")
                print(f"{escaper.substitute_html(str(row[1]))}<br>")
                a = row[0]
                b = row[1]
            if (a != row[0]) & (b != row[1]):
                print("</td>")
                print(f"</tr>")
                print("<tr><td colspan=2><hr></td></tr>")
                print(f"<tr>")
                print(f"<td style='text-align:center'><b>{escaper.substitute_html(str(row[0]))}</b></td>")
                print("<td>")
                print(f"{escaper.substitute_html(str(row[1]))}<br>")
                a = row[0]
                b = row[1]
            else:
                print(f"{escaper.substitute_html(str(row[1]))}<br>")
        print("</td>")
        print(f"</tr>")
        print("</table>")
        print("</center>")
        print("</td>")
        print("</table>")
        print("<br>")





    print('''
        <hr>
        <center>
                <table border=0 height="100px" width="60%">
                        <tr>
                                <td> <img src="/images/FIB-web.png" height="60%"> </td>
                                <td> <img src="/images/hackers-upc-web.png" height="60%"> </td>
                                <td> <img src="/images/bsc-web.png" height="60%"> </td>
                                <td> <img src="/images/plan-tl-web.png" height="60%"> </td>
                        </tr>
                </table>
        </center>
    ''')


    print("</html>")

if __name__ == "__main__":
    main()
