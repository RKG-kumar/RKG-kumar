import networkx as nx
import numpy as np
import pandas as pd
import math as mt
import queue as Qu
from collections import Counter
import matplotlib.pyplot as plt
import operator
from random import random
import math
import collections
from datetime import datetime
import glob
#nx.eigenvector_centrality(g, max_iter = 100000)
#[26, 25, 27, 24, 28, 23]

def find_list(D,nd,best_list):
    while(D[nd][5]!=-1):
        best_list.append(nd)
        nd = D[nd][5]
    best_list.append(nd)
    
def single_source_path(d,g,q,bst_list):
    
    while(q.empty()==False):
        nd = q.get()
       
        d[nd][3] = True

        neighbours = nx.neighbors(g,nd)

        pp = [x for x in neighbours]
        
        for i in pp:
            if d[i][4] == False and d[i][3]== False:
                q.put(i)
                ll = list(q.queue)
               
                d[i][4] = True
                
                d[i][1] = d[nd][1] + 1
                
                d[i][2] = d[nd][2] + d[i][0]
                d[i][5] = nd
                
            elif d[i][3]== False and d[nd][1] < d[i][1] and d[i][2] < d[nd][2] + d[i][0]:
                
                d[i][2] = d[nd][2] + d[i][0]
                d[i][5] = nd

    mx = -1
    nd =''
    sm = 0
   
    for i in d.keys():
        if mx <= d[i][1]: # added one  sm < d[i][2]
            mx = d[i][1]
            if sm < d[i][2]:
                nd = i
                sm = d[i][2]
              
    best_list = find_list(d,nd,bst_list)
    return best_list
        
def calculate_best_list(cent,g):
    
    D ={}
    mn = min(cent.values())
    for i in cent.keys():
        D[i] = [cent[i],0,cent[i],False,False,0]
        
        if mn == cent[i]:
            mn = i
   
    que = Qu.Queue(maxsize=len(cent.keys()))
    bst_list =[]
    que.put(mn)
   
    D[mn][4] = True
    D[mn][5] = -1
    single_source_path(D,g,que,bst_list)
    return bst_list
    
def maximum_components(g):
        p = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]
        md=-1
        m1 = g.copy()
        list_nd =[]
        list_nodes = []
        sm2 = 0
        for k in p:
            m = g.subgraph(list(k))
            #print(nx.diameter(m))
            #calculate centrality
            cent = nx.eigenvector_centrality(m, max_iter=100000)
            list_nodes = calculate_best_list(cent,m)
            
            # choose best one list 
            D = {}
            for i in list_nodes:
                D[i]= cent[i]
            list_nodes = []
            #print("Dictioney is {0}".format(D))
            for k in sorted(D, key=D.get, reverse=True):
                list_nodes.append(k)
            
            # take reverse order
            
            L1 = len(list_nodes)
            L2 = len(list_nd)
            
            if L2 == L1:
                sm1 = sum([cent[i] for  i in list_nodes])
                if sm1 > sm2:
                    list_nd = list_nodes
                    sm2 = sm1
            elif L2 < L1:
                list_nd = list_nodes
                sm2 = sum([cent[i] for  i in list_nodes])
                
        return list_nd,len(p)
  
def best_list(centrality,lk,G):
    avg = 0
    mn = min(centrality.values())
    mnd=''
    D={}
    for i in centrality.keys():
        if centrality[i]==mn:
            mnd=i
    for i in centrality.keys():
        t= [p for p in nx.all_shortest_paths(G, source=mnd, target=i)]
        for j in t:
            sm=0
            for k in j:
                sm = sm + centrality[k]
            #print(sm)
            if(avg < sm):
                avg =sm
                lk =j
    print("final list is : {0}".format(lk))
    return lk
def diffrence(x, y):
    if x >= y:
        result = x - y
    else:
        result = y - x
    return result
def median(lst):
    n = len(lst)
    s = sorted(lst)
    return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None

def list_eg(centrality,strtnd,REVERSE):
    mx = max(centrality.values())
    mn = min(centrality.values())
    D = {}
    srtnd =[]
    #srtnd.append(mx)
    #D[mx]=centrality[mx]
    
    ls = [(node,centrality[node]) for node in centrality]
    for i in ls:
        diff1 = diffrence(mx, i[1])
        diff2 = diffrence(mn, i[1])
        if diff1 <= diff2:
            srtnd.append(i[0])
            
    for i in srtnd:
        D[i]= centrality[i]
    srtnd=[]
    #print("Dictioney is {0}".format(D))
    for k in sorted(D, key=D.get, reverse=REVERSE):
        srtnd.append(k)
    return srtnd
 def R_bestNode(srtnd):
    return sorted(srtnd,reverse=False)

def bestNode_list(srtnd,g,r):
    if(len(srtnd)==0):
        return False
    nd= srtnd[0]    
    m=nx.ego_graph(g,nd, radius=r, undirected=True)
    li=[str(x) for x in m]
    sz= len(li)
    lk = []
    rtnd =''
    # nlist =[]
    nlist = []
    ndict = {} 
    for  i in  srtnd:
        m=nx.ego_graph(g,i, radius=r, undirected=True)
        li=[x for x in m]
        
        tg = g.copy()
        tg.remove_nodes_from(li)
        l1 = len(tg.nodes())-r
        l2 = len(tg.edges())
        sz1 = len(li)
        if sz <= sz1 and l1 <= l2:
            sz = sz1
            rtnd = i
            lk=li
            
        if l1 <= l2:
            ndict[i] = li
            nlist.append(i)
            #print("ith node number= {3} total_node = {0} and edge = {1} radius = {2}".format(l1,l2,r,i))
        
    #print(lk)
    #p = [c for c in sorted(nx.connected_components(tg), key=len, reverse=True)]
    #print("number of connected components after taking maximum = {0} with {1} ".format(len(p),rtnd))
    print("nlist = {0} radius = {1}".format(nlist,r))
    return nlist,ndict
    
 def diffrence(x, y):
    if x >= y:
        result = x - y
    else:
        result = y - x
    return result

def heavy_nodes(g):
    p = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]  
    D = {}
    #for k in p:
        #m = g.subgraph(list(k))
        #cent = nx.eigenvector_centrality(m, max_iter=100000)
        #for i in cent.keys():
            #D[i] = cent[i]
    D = nx.eigenvector_centrality(g, max_iter=100000)
    return D

def get_group_list(g):
    D  = heavy_nodes(g)
    mx = max(D.values())
    mn = min(D.values())
    srtnd = []
    for i in D.keys():
        if(diffrence(mx, D[i]) <= diffrence(mn, D[i])):
            srtnd.append(i)
    d = {}
    for i in srtnd:
        d[i] = D[i]
    #print(srtnd)
    srtnd = []
    for k in sorted(d, key=d.get, reverse=True):
        srtnd.append(k)
    return srtnd

def bestNode_in_group(g,r):
    #print(srtnd)
    srtnd = get_group_list(g)
    nd= srtnd[0]
    m=nx.ego_graph(g,nd, radius=r, undirected=True)
    li=[str(x) for x in m]
    sz= len(li)
    rtnd =''
    for  i in  srtnd:
        m=nx.ego_graph(g,i, radius=r, undirected=True)
        li=[str(x) for x in m]
        sz1 = len(li)
        if sz <= sz1:
            sz = sz1
            rtnd = i
    #print("number of remaining edges ={0} number of nodes = {}")
    return rtnd
#bestNode_in_group(g,6)
def take_decrease_order(li,d):
    D = {}
    for i in li:
        if i != -1:
            D[i] = d[i]
    srtnd = []
    for k in sorted(D, key=D.get, reverse=False):
        srtnd.append(k)
        #print(k,D[k])
    return srtnd
    
def best_list_node(g,r):
    cent = nx.eigenvector_centrality(g, max_iter=100000)
    srtnd = []
    for k in sorted(cent, key=cent.get, reverse=True):
        srtnd.append(k)
    m=nx.ego_graph(g,1912, radius=3, undirected=True)
    li=[x for x in m]
    first = 0
    last  = len(srtnd)
    mx1   = len(li)
    mx2   = 0
    while(first+1 != last):
        mid = mt.floor((first + last)/2)
        m=nx.ego_graph(g,srtnd[mid], radius=r, undirected=True)
        li=[str(x) for x in m]
        if len(li)>= mx1:
            mx1 = len(li)
            first = mid
        else:
            last = mid
        print(first,mid,last,len(li),mx1) 
    return srtnd[last]

def best_node_cmp(g,G,r):        
    #G = g.copy() 
    d = G.degree()
    cent = nx.eigenvector_centrality(g, max_iter=100000)
    mx1 = max(cent , key=cent.get)
    m = nx.ego_graph(g,mx1, radius=r, undirected=True)
    li = [x for x in m]
    g.remove_nodes_from(li)
    #print(len(li))
    p = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    #print("Number of components = {0}".format(len(p)))
    li = []
    for k in p:
        m = g.subgraph(list(k))
        #cent = nx.eigenvector_centrality(m, max_iter=100000)
        #mx = min(cent , key=cent.get)
        ncent = {}
        for n in m.nodes():
            ncent[n] = cent[n]
        mx = min(ncent , key=ncent.get)    
        #print("inside for loop 1!")
        if nx.has_path(G,mx,mx1) == True:
            #nx.draw(m,with_labels=True, font_weight='bold',node_size=1000)
            #plt.show()
            #print("yes path "+str(mx)+" , "+str(mx1))
            path = [q for q in nx.shortest_path(G, source=mx, target=mx1)]
            #print(path)
        else:
            continue
        #for j in path:
            #print("inside for loop 1.5 !")
        km = take_decrease_order(path,d)
        ln = len(km)
        while ln < r:
            km.append(-1)
            ln = len(km)
            #print("inside loop 1.7 !")
        li.append(km[-r:])
        #print("inside for loop 1.6 !")
        #print(len(km[-r:]),r)
        #for _ range(0,r-len(li)):
            
    t = 0
    k = mx1
    c = np.array(li)
    #print(c)
    if len(li)==0:
        #print("SINGLE node burn all for r ="+str(r))
        return k
    for j in range(0,r):
        #print("inside for loop 2 !"+ str(c)+ " "+str(j))
        li = set(c[:,j])
        li = take_decrease_order(li,d)
        for i in li[0:r]:
            m=nx.ego_graph(G,i, radius=r, undirected=True)
            li1 = [x for x in m]
            if t <= len(li1):
                t = len(li1)
                k = i

    #print(set(li),k,t)
    #print("Function completed !")
    return k

def printBGR(BGR):
    r = 0
    for key in BGR.keys():
        print("Radius = "+str(key)+" Nodes = "+ str(BGR[key][0]))
def cornerHuerestic(g1,G,r,BGR,fixed):
    max_gues = r
    listg = range(0,max_gues)
    listg= sorted(listg, reverse=True)
    #print("FIXED = "+str(fixed))
    #g1 = G.copy()
    for i in listg:
        if i == fixed :
            continue
        Gc = max(nx.connected_components(g1), key=len)
        m  = g1.subgraph(list(Gc))
        g  = nx.Graph(m)
        nd = best_node_cmp(g,G,i)
        #print("Node = {0} and Radius = {1}".format(nd,i))
        m  = nx.ego_graph(G,nd, radius=i, undirected=True)
        li = [x for x in m]
        BGR[i] = [nd,li]
        g1.remove_nodes_from(li)
        #remove_more(g1,G,BGR)
        #print(len(li))
        if len(g1.nodes())==0:
            #print("ALL NODES are burned by burning number {0}".format(r))
            #printBGR(BGR)
            break

def C_overlape(G,g,BGR,r):
    #print("Looking for overlape Bruning Number = "+str(r))
    cent = nx.eigenvector_centrality(G, max_iter=100000)
    mn = 1000
    ndm = -1
    for nodes in g.nodes():
        #print(cent[nodes])
        if mn > cent[nodes]:
            mn  =  cent[nodes]
            ndm = nodes 
    
    p = nx.single_source_shortest_path(G,ndm)
    S = {} 
    for rng in range(0,r):
        #print(p[BGR[rng][0]])
        if BGR[rng][0] in  p.keys() and (len(p[BGR[rng][0]]) - rng) <= rng:
            S[rng] = sorted(p[BGR[rng][0]] , reverse=True)
    #print(S)
    for keys in S.keys():
        for nodes in S[keys]:
            g = G.copy()
            m = nx.ego_graph(g,nodes,radius = keys, undirected=True)
            li1=[x for x in m]
            g.remove_nodes_from(li1)
            #burn_graph(g,li,BGR,G,keys)
            cornerHuerestic(g,G,r,BGR,keys)
            if len(g.nodes()) ==0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                #print("END TIME =", current_time)
                print("ALL NODES are burned by burning number {0}".format(r))
                #printBGR(BGR)
                print("START TIME =", current_time)
                #print("END TIME =", current_time1)
                
                return 1


    
    
def cornerBin(g,G,mx):
    i = 1
    j =  mx
    while i <= j:
        #current_time()
        r = int( ( i + j )/2 )
        g = G.copy()
        BGR = {}
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        #print("START  TIME =", current_time)
        cornerHuerestic(g,G,r,BGR,-1)
        if len(g.nodes())!= 0:
            #adjust_circles(g,G,BGR,r)
            #print(g.nodes())
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            #print("END TIME =", current_time)
            C_overlape(G,g,BGR,r)
            #print(g.nodes())
                    #current_time()
            if len(g.nodes())!= 0:
                i = r + 1
        else :
            now = datetime.now()
            current_time1 = now.strftime("%H:%M:%S")
            #print("END TIME =", current_time)
            print("ALL NODES are burned by burning number {0}".format(r))
            #printBGR(BGR)
            print("START TIME =", current_time)
            print("END TIME =", current_time1)
            if i == j:
                break
            else :
                j = r - 1
                
def bestNode(srtnd,g,r):
    #print(srtnd)
    #print("\\textbf{Longest with maximum centrality sum } = "+str(srtnd))
    if len(srtnd)==0:
        return -1
    nd= srtnd[0]
    m=nx.ego_graph(g,nd, radius=r, undirected=True)
    li=[str(x) for x in m]
    sz= len(li)
    rtnd =''
    for  i in  srtnd:
        m = nx.ego_graph(g,i, radius=r, undirected=True)
        li=[str(x) for x in m]
        sz1 = len(li)
        if sz <= sz1:
            sz = sz1
            rtnd = i
    #print("number of remaining edges ={0} number of nodes = {}")
    return rtnd
    
def current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    #print("Time =", current_time)
    
def create_best_list(g,G,BGR):
    newd = max(BGR, key=BGR.get)
    li = []
    k = g.nodes()
    for i in k:
        if nx.has_path(G,i,newd):
            t= [p for p in nx.all_shortest_paths(G, source=i, target=newd)]
            for j in t:
                li.append(j[-2])

    li = set(li)
    li = list(li)
    cent = nx.degree(G)
    D = {}
    for i in li:
        D[i] = cent[i]
        
    #print(D)
    li = []
    for k in sorted(D, key=D.get, reverse=True):
        li.append(k)
    return li

    
        
def burn_graph(g,itli,BGR,G,fixed):
    for i in itli:
        srtnd=[]
        lk= len (list(g.nodes()))
        if i == fixed:
            continue
        if lk == 0:
            break
        srtnd,c_comp = maximum_components(g)
        nd = bestNode(srtnd,g,i)
       # print("Ridius "+str(i)+ "node = "+str(nd))
        m=nx.ego_graph(G,nd, radius=i, undirected=True)
        li=[x for x in m]
        BGR[i] = [nd , li]
        #print(len(li))
        g.remove_nodes_from(li)

def adjust_circles(g,G,BGR,r):
    li = create_best_list(g,G,BGR)
    if len(li)==0 and len(G.nodes()) == 0:
        print("All nodes have burned !")
    for i in li[0:r]:
        g = G.copy()
        m=nx.ego_graph(g,i,radius=r-1, undirected=True)
        li1=[x for x in m]
        #print("best node is {0} of radius {1}".format(i,r-1))
        BGR= {}
        BGR[i] = r-1
        g.remove_nodes_from(li1)
        li = [x for x in range(0,r-1)]
        li = sorted(li, reverse=True)
        burn_graph(g,li,BGR,G,-1)
        if len(g.nodes())==0:
            print("best node is {0} of radius {1}".format(i,r-1))
            print("graph was burned with burning number = {0}".format(r))
            break
        else:
            print("remaining nodes are {0}".format(len(g.nodes())))


def overlape(G,g,BGR,r):
    #print("Looking for overlape Bruning Number = "+str(r))
    cent = nx.eigenvector_centrality(G, max_iter=100000)
    mn = 1000
    ndm = -1
    for nodes in g.nodes():
        #print(cent[nodes])
        if mn > cent[nodes]:
            mn  =  cent[nodes]
            ndm = nodes 
    
    p = nx.single_source_shortest_path(G,ndm)
    S = {} 
    for rng in range(0,r):
        #print(p[BGR[rng][0]])
        if BGR[rng][0] in  p.keys() and (len(p[BGR[rng][0]]) - rng) <= rng:
            S[rng] = sorted(p[BGR[rng][0]] , reverse=True)
    #print(S)
    for keys in S.keys():
        for nodes in S[keys]:
            g = G.copy()
            m = nx.ego_graph(g,nodes,radius = keys, undirected=True)
            li1=[x for x in m]
            g.remove_nodes_from(li1)
            li = [x for x in range(0,r)]
            li = sorted(li, reverse=True)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            #print("START  TIME =", current_time)
            burn_graph(g,li,BGR,G,keys)
            if len(g.nodes()) ==0:
                now = datetime.now()
                current_time1 = now.strftime("%H:%M:%S")
                #print("END TIME =", current_time)
                print("ALL NODES are burned by burning number {0}".format(r))
                print("START  TIME =", current_time)
                print("END  TIME =", current_time1)
                return 1
            
        
    #print("- : Graph is not Burned :-")
    # [4957, 4580, 2494, 5493, 5748, 5800, 5016, 1474]
    
 def MAINFUNCT(g,G,mx):
    i = 1
    j =  mx
    while i <= j:
        #current_time()
        r = int( ( i + j )/2 )
        g = G.copy()
        BGR = {}
        li = [x for x in range(0,r)]
        li = sorted(li, reverse=True)
        #print(li)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        #print("START  TIME =", current_time)
        burn_graph(g,li,BGR,G,-1)
        #current_time()
        if len(g.nodes())!= 0:
            #adjust_circles(g,G,BGR,r)
            #print(g.nodes())
            overlape(G,g,BGR,r)
            #print(g.nodes())
            #current_time()
            if len(g.nodes())!= 0:
                i = r + 1
            else:
                print("Burn by overlape!")
        else :
            now = datetime.now()
            #current_time = now.strftime("%H:%M:%S")
            #print("END TIME =", current_time)
            print("ALL NODES are burned by burning number {0}".format(r))
            now = datetime.now()
            current_time1 = now.strftime("%H:%M:%S")
            print("START  TIME =", current_time)
            print("END  TIME =", current_time1)
            #if i == j:
                #break
            #else :
            j = r - 1 
           
# make function for component graph [3525, 2036, 1761, 1673, 697, 603, 211] [3525, 1673, 2036, 603, 1761, 697, 211]
# for component graph 
GBGR ={}
def calculate_burning_num(G,g1,r,GID):
    list_g = sorted(range(0,r+1),reverse=True)
    bg_n = -1
    first = 0
    last = r+1
    for i in list_g:
        #print("IT IS IN PROGRESS.. > "+str(i))
        g2 = g1.copy()
        BGR ={}
        for j in range(0,i):
            p = [c for c in sorted(nx.connected_components(g2), key=len, reverse=True)]
            #print("Number of components = {0}".format(len(p)))
            li = []
            mx = -1
            newg = g2.copy()
            if(len(p) > 1):
                for k in p:
                    m  = g2.subgraph(list(k))
                    keys = GID.keys()     
                    m = nx.Graph(m,idx=-1)
                    #print(m.nodes())
                    bg = -1
                    for key in GID.keys():
                        if Counter(list(m.nodes())) == Counter(GID[key][1]):
                            #print(m.nodes(),GID[key][1])
                            bg = GID[key][0]
                    if bg == -1:
                        #print("Called")
                        bg,mt,gmt  = calculate_burning_num(G,m,i-1,GID)
                        if len(GID.keys())==0:
                            GID[0] = [bg,list(m.nodes())]
                            print("Started ...")
                        else:
                            key = max(GID.keys()) + 1
                            GID[key] = [bg,list(m.nodes())]
                            print(key)
                        
                    #print("bg = {0} of nodes = {1}".format(bg,m.nodes()))
                    if(bg > mx):
                        mx   = bg
                        newg = m
                    #break
                        
            #pos = nx.spring_layout(newg,k=0.15,iterations=20)
           # nx.draw(newg,layout=pos,with_labels=True, font_weight='bold',node_size=1000)
            #plt.show()
            #print("end")
            srtnd,c_comp = maximum_components(newg)
            #print("Selected Graph = {0}".format(newg.nodes()))
            nd = bestNode(srtnd,newg,i-j-1)
            
            if nd == -1:
                return -1
            #nd = bestNode_in_group(g,r-i)
            #print(i,j,i-j-1)
            BGR[nd] = i-j-1
            #remove_more(g2,G,BGR)
            #print(g2.nodes())
            #print("best node is {0} of radius {1}".format(nd,i-j-1))
            m = nx.ego_graph(G,nd, radius=i-j-1, undirected=True)
            li=[x for x in m]
            #print(len(li))
            g2.remove_nodes_from(li)
            if len(g2.nodes())==0:
                bg_n = i
                break
        if len(g2.nodes()) !=0:
            GBGR = BGR
            #print(BGR,len(g2.nodes()))
            return bg_n,BGR,g2

textfiles = []
for file in glob.glob("DATA/TestDATA/*.txt"):
    print("* -----**********---------- *")
    print(file)
    print("* -----**********---------- *")
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    #print("STARTING Time =", current_time)
    g = nx.read_adjlist(file, nodetype=int)
     # 20 , 5 
    # GENERATE TREEE
    # 1000 12
    #g = nx.random_tree(20, seed=4)
    #pos = nx.spring_layout(g,k=0.15,iterations=20)
    #nx.draw(g,layout=pos,with_labels=True, font_weight='bold',node_size=1000)
    #plt.show()
    # GENERATE TREEE
    
    
    G = g.copy()
    print("******* - Corner Heurestics - ******** ")
    cornerBin(g,G,600)
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    #print("STARTING Time =", current_time)
    print("******* - Backbone Based Greedy - ******** ")
    MAINFUNCT(g,G,600)
    
    #print("******* - TREE ALGORITHM - ******** ")
    #TreeAlgo(g,G)
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    #print("END TIME =", current_time)
    #break
    
    ## component base 
    #G = nx.read_adjlist(file, nodetype=int)
    g1 = G.copy()
    GID ={}
    BG = 500 # bg number
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("START  TIME =", current_time)

    bg,GBGR,g2 = calculate_burning_num(G,g1,BG,GID)
    #print(len(g1.nodes()),GBGR,g2.nodes())
    print("******* - RECURSIVE BACKBONE BASED GREEDY - ******** ")
    print("Burning of graph = {0}".format(bg))
    #adjust_circles(g2,G,GBGR,BG)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("START  TIME =", current_time)
    
