import numpy as np
import networkx as nx
import itertools
import os
from math import comb
from string import ascii_lowercase as alc
import pickle 
import requests
import time




def reduce(a,v):
    reduce = True
    cur_a = a.copy()
    while reduce:
        # compute current degree vector
        d = cur_a.sum(axis=-1)
        # check if there is a valid node to reduce
        # reduction_step = np.logical_and(np.logical_or(d==1,d==2),1-v)
        reduction_step = np.logical_and(np.logical_or(d==1,d==2),v<1)
        remove_node = np.argwhere(reduction_step)
        if remove_node.size == 0:
            # no node to reduce. 
            # computable = np.all((d>0) == (v==1)) or np.all(d==0)
            computable = np.all((d>0) == (v>=1)) or np.all(d==0)
            return computable
        else:
            remove_node = remove_node[0]
            if d[remove_node] == 2:
                # replace node with edge
                neighbors = np.argwhere(a[remove_node,:].squeeze())
                cur_a[neighbors[0], neighbors[1]] = 1
                cur_a[neighbors[1], neighbors[0]] = 1
            # remove node
            cur_a[remove_node,:] = 0
            cur_a[:, remove_node] = 0

def comp_nodes(v_dict,u_dict):
    return v_dict['feature'] == u_dict['feature']

def get_one_dict(num_nodes):
    # get all possible options for single red node
    dict_arr = []
    for i in range(num_nodes):
        d = {}
        for j in range(num_nodes):
            if j == i:
                d[j] = 1
            else:
                d[j] = 0
        dict_arr.append(d)
    return dict_arr

def get_two_dict(num_nodes):
    # get all possible options for two red node
    dict_arr = []
    all_combos = list(itertools.combinations([i for i in range(num_nodes)], 2))
    for combo in all_combos:
        d_1 = {}
        d_2 = {}
        first_index = True
        for j in range(num_nodes):
            if j in combo:
                if first_index:
                    d_1[j] = 1
                    d_2[j] = 2
                    first_index = False
                else:
                    d_1[j] = 2
                    d_2[j] = 1
            else:
                d_1[j] = 0
                d_2[j] = 0
        dict_arr.append(d_2)
        dict_arr.append(d_1)
        # d = {}
        # for j in range(num_nodes):
        #     if j in combo:
        #         d[j] = 1
        #     else:
        #         d[j] = 0
        # dict_arr.append(d)
    return dict_arr

def dict_to_vec(dict):
    arr = []
    for i in range(len(dict)):
        arr.append(dict[i])
    return np.array(arr)

def filter_isomorphism(g, color_dict):
    # given a graph and coloring options reduce ismorphic copies
    g_copy = [g.copy() for _ in range(len(color_dict))]
    filter_color = []
    filter_g = []
    for i, g_tmp in enumerate(g_copy):
        c_dict = color_dict[i]
        nx.set_node_attributes(g_tmp,c_dict, 'feature')
        add = True
        for test_g in filter_g:
            if nx.is_isomorphic(g_tmp,test_g, node_match=comp_nodes):
                add = False
                break
        if add:
            filter_g.append(g_tmp)
            filter_color.append(dict_to_vec(c_dict))
    return filter_color



def get_ppgn_additional_subgraphs(path, num_edges):
    print('compute {} edges'.format(num_edges))
    g_path = os.path.join(os.path.join(path,'ge{}c.g6'.format(num_edges)))
    G_list = nx.read_graph6(g_path)
    res_arr = []
    poly_count = 0
    i=0
    for g in G_list:
        num_nodes = g.number_of_nodes()
        color_list = []
        one_red = filter_isomorphism(g,get_one_dict(num_nodes))        
        two_red = filter_isomorphism(g,get_two_dict(num_nodes))        
        a = nx.to_numpy_array(g)
        compute = reduce(a,np.zeros(num_nodes))
        if not compute:
            color_list.append(np.zeros(num_nodes))
        for list in [one_red, two_red]:
            for v in list:
                compute = reduce(a,v)
                if not compute:
                    color_list.append(v)
        if len(color_list) > 0:
            poly_count += len(color_list)
            res_arr.append((a,color_list))
    print('number of non-computable graphs with {} edges is {} ({})'.format(num_edges, len(res_arr), poly_count))
    with open(os.path.join(path,'ppgn_{}.npy'.format(num_edges)), 'wb') as f:
            np.save(f,res_arr)

def iso_check(g_arr, c):
    filter_g = []
    for g in g_arr:
        add = True
        for g_test in filter_g:
            if nx.is_isomorphic(g,g_test, node_match=comp_nodes):
                add = False
                break
        if add:
            filter_g.append(g)
    res = [(nx.to_numpy_matrix(g), [c]) for g in filter_g]
    return res

def parse_poly_file_to_string(path):
    adj_color_list = np.load(path, allow_pickle=True)
    string_list = []
    for a, c_list in adj_color_list:
        for c in c_list:
            string_list.append(string_poly_from_adj(a,c))
    return string_list

def convert_ppgn_polynomials_to_strings(main_path, d):
    string_list = []
    string_list += parse_poly_file_to_string(os.path.join(main_path,'ppgn_{}.npy'.format(d)))
    print('{} degree polynomials - total {} polynomials'.format(d, len(string_list)))
    pickle.dump(string_list, open( os.path.join(main_path,'ppgn_{}.pkl'.format(d)), "wb" ) )

def string_poly_from_adj(a,c):
    poly=''
    index_dict = {}
    n = a.shape[0]
    m = int(a.sum()/2 + a.trace()/2)
    pol_count = 0
    for t, char in enumerate(alc):
        index_dict[t] = char
        if t == (n-1):
            break
    for i in range(n):
        for j in range(i,n):
            if a[i,j] > 0:
                for _ in range(int(a[i,j])):
                    pol_count += 1
                    if pol_count == m:
                        poly += 'k'+index_dict[i]+index_dict[j]+' -> '
                    else:
                        poly += 'k'+index_dict[i]+index_dict[j]+', '
    red_sum = c.sum()
    if red_sum == 0:
        poly += 'kyz'
    elif red_sum == 1:
        poly += 'k'+index_dict[np.argwhere(c)[0][0]] + 'z'
    else:
        poly += 'k'+ index_dict[np.argwhere(c).squeeze()[0]] + index_dict[np.argwhere(c).squeeze()[1]]
    return poly

def remove_invariant_poly(d):
    new_arr = []
    name = os.path.join('data','graph_data','{}.pkl'.format(d))
    poly_list = pickle.load(open(name,'rb'))
    for poly in poly_list:
        if not poly.endswith('yz'):
            new_arr.append(poly)
    pickle.dump(new_arr, open( os.path.join('data','graph_data','{}.pkl'.format(d)), "wb" ) )
    print('number of poly {}'.format(len(new_arr)))
    
def download_data(path):
    URL = "https://users.cecs.anu.edu.au/~bdm/data/ge3c.g6"
    response = requests.get(URL)
    open(os.path.join(path,"ge3c.g6"), "wb").write(response.content)
    URL = "https://users.cecs.anu.edu.au/~bdm/data/ge4c.g6"
    response = requests.get(URL)
    open(os.path.join(path,"ge4c.g6"), "wb").write(response.content)
    URL = "https://users.cecs.anu.edu.au/~bdm/data/ge5c.g6"
    response = requests.get(URL)
    open(os.path.join(path,"ge5c.g6"), "wb").write(response.content)
    URL = "https://users.cecs.anu.edu.au/~bdm/data/ge6c.g6"
    response = requests.get(URL)
    open(os.path.join(path,"ge6c.g6"), "wb").write(response.content)

def get_ppgn_uncomputable_equivariant_polynomials(path):
    start = time.time()
    for d in [5,6]:
        get_ppgn_additional_subgraphs(path, d)
        convert_ppgn_polynomials_to_strings(path, d)
        # remove_invariant_poly(d)
    print('time to complete collecting all non-computable polynomials up to degree 7: {}'.format(time.time()-start))

if __name__ == "__main__":
    src_path = os.getcwd()
    path = os.path.join(src_path, 'data/equivaraint_polynomials')
    download_data(path)
    start = time.time()
    get_ppgn_uncomputable_equivariant_polynomials(path)