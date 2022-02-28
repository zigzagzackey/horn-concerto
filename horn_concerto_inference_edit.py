#!/usr/bin/env python
"""
Horn Concerto - Inference from Horn rules.
Author: Tommaso Soru <tsoru@informatik.uni-leipzig.de>
Version: 0.0.7
Usage:
    Use test endpoint (DBpedia)
    > python horn_concerto_inference.py <endpoint> <graph_IRI> <rules_PATH> <infer_function> <output_folder>
"""
import urllib.request, urllib.error, urllib.parse, urllib.request, urllib.parse, urllib.error, http.client, json
import sys
import os
import codecs
import pickle
import time
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import importlib

importlib.reload(sys)
# sys.setdefaultencoding("utf-8")

VERSION = "0.0.7"

endpoint = None
graph = None
rules = None
infer_fun = None
output_folder = None

############################### FUNCTIONS ################################

def sparql_query(query):
    param = dict()
    param["default-graph-uri"] = graph
    param["query"] = query
    param["format"] = "JSON"
    param["CXML_redir_for_subjs"] = "121"
    param["CXML_redir_for_hrefs"] = ""
    param["timeout"] = "600000" # ten minutes - works with Virtuoso endpoints
    param["debug"] = "on"
    try:
        resp = urllib.request.urlopen(endpoint + "?" + urllib.parse.urlencode(param))
        j = resp.read()
        resp.close()
    except (urllib.error.HTTPError, http.client.BadStatusLine):
        print("*** Query error. Empty result set. ***")
        print("*** {}".format(query))
        j = '{ "results": { "bindings": [] } }'
    sys.stdout.flush()
    return json.loads(j)

def opposite_product(a):
    return 1 - np.prod(np.ones(len(a)) - a)

files = ["pxy-qxy", "pxy-qyx", "pxy-qxz-rzy", "pxy-qxz-ryz", "pxy-qzx-rzy", "pxy-qzx-ryz"]

def retrieve(t, predictions):
    
    global files
    preds = dict()
    
    with open(rules + r"/rules-" + files[t] + ".tsv") as f:
        next(f)
        for line in f:
            line = line[:-1].split('\t')
            weight = float(line[0])
            head = line[1]
            body = list()
            for i in range(len(line[3:])//2):
                body.append((line[3+i*2], line[4+i*2][1], line[4+i*2][3]))
            # print head, body
            bodies = ""
            for b in body:
                bodies += "?{} <{}> ?{} . ".format(b[1], b[0], b[2])
            offset = 0
            while True:
                query = "SELECT DISTINCT(?x) ?y WHERE { " + bodies + "MINUS { ?x <" + head + "> ?y } } LIMIT 10000 OFFSET " + str(offset)
                print(query)
                results = sparql_query(query)
                print(len(results["results"]["bindings"]))
                # print "\t", results
                try:
                    for result in results["results"]["bindings"]:
                        triple = "<{}> <{}> <{}>".format(str(result["x"]["value"]), head, str(result["y"]["value"]))
                        # print weight, triple
                        if triple not in preds:
                            preds[triple] = list()
                        preds[triple].append(weight)
                        # print predictions[triple]
                except KeyError:
                    pass
                if len(results["results"]["bindings"]) == 10000:
                    offset += 10000
                else:
                    break
        
    return preds

############################### ALGORITHM ################################

def run(endpoint_P, graph_P, rules_P, infer_fun_P, output_folder_P):
    
    global endpoint, graph, rules, infer_fun, output_folder
    global files
    
    endpoint = endpoint_P
    graph = graph_P
    rules = rules_P
    infer_fun = infer_fun_P
    output_folder = output_folder_P
    
    print("Horn Concerto v{}".format(VERSION))
    print("Endpoint: {}\nGraph: {}\nRules: {}\nInference function: {}\nOutput folder: {}\n".format(endpoint, graph, rules, infer_fun, output_folder))
    num_cores = multiprocessing.cpu_count()
    print("Cores:", num_cores)
    
    # WARNING: temporary in-memory solution
    predictions = dict()

    print("Retrieving conditional probabilities...")
    preds = Parallel(n_jobs=num_cores)(delayed(retrieve)(t=t, predictions=predictions) for t in range(len(files)))

    for p in preds:
        for triple in p:
            if triple not in predictions:
                predictions[triple] = list()
            for val in p[triple]:
                predictions[triple].append(val)
    if os.name == 'nt':
        with codecs.open("{}/predictions.txt".format(output_folder), 'w', 'cp932', 'ignore') as fout:
            for p in predictions:
                fout.write("{}\t{}\n".format(p, predictions[p]))
    else:
        with open("{}/predictions.txt".format(output_folder), 'w') as fout:
            for p in predictions:
                fout.write("{}\t{}\n".format(p, predictions[p]))

    print("Computing inference values...")
    for fun in infer_fun.split(","):
    
        predictions_fun = dict()

        for triple in predictions:
            if fun == 'A':
                predictions_fun[triple] = np.mean(predictions[triple])
            if fun == 'M':
                predictions_fun[triple] = np.max(predictions[triple])
            if fun == 'P':
                predictions_fun[triple] = opposite_product(predictions[triple])

        print("Number of predicted triples:", len(predictions_fun))
        print("Saving predictions to file...")

        if os.name == 'nt':
            with codecs.open("{}/inferred_triples_{}.txt".format(output_folder, fun), "w", 'cp932', 'ignore') as fout:
                for key, value in sorted(iter(predictions_fun.items()), key=lambda k_v: (k_v[1], k_v[0]), reverse=True):
                    # print "%.3f\t%s" % (value, key)
                    fout.write("%.3f\t%s\n" % (value, key))
        else:
            with open("{}/inferred_triples_{}.txt".format(output_folder, fun), "w") as fout:
                for key, value in sorted(iter(predictions_fun.items()), key=lambda k_v: (k_v[1],k_v[0]), reverse=True):
                    # print "%.3f\t%s" % (value, key)
                    fout.write("%.3f\t%s\n" % (value, key))

if __name__ == '__main__':
    
    ############################### ARGUMENTS ################################

    # endpoint = sys.argv[1]]
    # endpoint = "https://linkeddata.rcgs.jp/"

    # endpoint = "http://dbpedia.org/sparql"
    # graph = "http://dbpedia.org"

    # endpoint = "https://ja.dbpedia.org/sparql"
    # graph = "https://ja.dbpedia.org"

    endpoint = "https://sparql.linkedopendata.cf"
    # graph = "https://www.mirko.jp/ogura/LOD/ogura"
    graph = "https://www.mirko.jp/ogura/LOD/kajin"

    rules = r"."
    # infer_fun = 'M' # 'A' (average), 'M' (maximum), 'P' (opp.product)
    infer_fun = 'M'  # 'A' (average), 'M' (maximum), 'P' (opp.product)
    output_folder = "."

    run(endpoint, graph, rules, infer_fun, output_folder)
